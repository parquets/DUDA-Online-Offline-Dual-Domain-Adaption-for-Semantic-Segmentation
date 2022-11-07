from cProfile import label
import os
from random import random
from xml.dom.expatbuilder import parseString
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast as autocast
import torch.distributed as dist
from utils.metric import intersectionAndUnionGPU
import copy
from PIL import Image
import time
import random as rd
import math


class UDATrainer:
    def __init__(self, config, scaler, seg_model, 
                 seg_optimizer, seg_scheduler, 
                 source_seg_loss=None, target_seg_loss=None,
                 dis_model=None, dis_optimizer=None, 
                 adv_loss=None, dis_scheduler=None, 
                 pseudo_selector=None, gpu_id=0) -> None:
        super().__init__()
        self.config = config
        self.scaler = scaler
        self.seg_model = seg_model
        self.seg_optimizer = seg_optimizer
        self.seg_scheduler = seg_scheduler
        self.dis_model = dis_model
        self.dis_optimizer = dis_optimizer
        self.adv_loss = adv_loss
        self.dis_scheduler = dis_scheduler
        self.seg_loss = nn.CrossEntropyLoss(ignore_index=255)
        self.source_seg_loss = source_seg_loss
        self.target_seg_loss = target_seg_loss
        
        self.alpha_confident_bank = np.zeros(19)
        self.beta_confident_bank = np.zeros(19)

        self.pseudo_selector = pseudo_selector

        if config.train.resume_from:
            self.seg_model.cuda(gpu_id)
            self.seg_model.load_pretrained(config.train.resume_from, gpu_id)

        self.mean_model = None
        if config.use_ema:
            self.mean_model = torch.optim.swa_utils.AveragedModel(self.seg_model, device=gpu_id)
            self.mean_model.update_parameters(self.seg_model)
            assert self.is_equal(self.seg_model, self.mean_model)

        if config.use_distill:
            self.mean_model = copy.deepcopy(seg_model)
            self.mean_model.cuda(gpu_id)
            self.mean_model.load_pretrained(config.train.mean_resume_from, gpu_id)
        
            
        self.consist_loss = nn.MSELoss()
        self.alpha = config.pseudo.alpha_start
        self.beta = 0.25
        self.offline_update_start = 10000
        self.pseudo_selector = pseudo_selector
        self.gpu_id = gpu_id
        self.current_iteration = 0

    def is_equal(self, model, mean_model):
        unequal_key = []
        for key in mean_model.state_dict().keys():
            if key == 'n_averaged':
                continue
            _key = key.replace('module.', '')
            if torch.any(model.state_dict()[_key]!=mean_model.state_dict()[key]):
                unequal_key.append(key)
        print(str(len(unequal_key))+" keys are not equal!")
        return len(unequal_key)==0

    def get_lr(self):
        lr = self.seg_optimizer.param_groups[-1]['lr']
        return lr

    def train(self):
        self.seg_model.train()
        if self.dis_model is not None:
            self.dis_model.train()
        if self.mean_model is not None:
            self.mean_model.train()

    def eval(self):
        self.seg_model.eval()
        if self.dis_model is not None:
            self.dis_model.eval()
        if self.mean_model is not None:
            self.mean_model.eval()

    def convert_sync_batchnorm(self):
        self.seg_model = nn.SyncBatchNorm.convert_sync_batchnorm(self.seg_model)
        if self.dis_model is not None:
            self.dis_model = nn.SyncBatchNorm.convert_sync_batchnorm(self.discriminator)
    
    def model_ddp(self):
        self.seg_model = nn.parallel.DistributedDataParallel(self.seg_model, device_ids=[self.gpu_id])
        if self.dis_model is not None:
            self.dis_model = nn.parallel.DistributedDataParallel(self.discriminator, device_ids=[self.gpu_id])
             
    def optimizer_zero_grad(self):
        if self.seg_optimizer is not None:
            self.seg_optimizer.zero_grad()
        if self.dis_model is not None:
            self.dis_optimizer.zero_grad()

    def optimizer_step(self):
        if self.seg_optimizer is not None:
            self.seg_optimizer.step()
        if self.dis_model is not None:
            self.dis_optimizer.step()
    
    def schedule_step(self):
        self.seg_scheduler.step()
        if self.dis_model is not None:
            self.dis_scheduler.step()

    def adjust_alpha(self, power=1.1):
        if self.current_iteration < 2000:
            self.alpha = self.config.pseudo.alpha_start 
        else: 
            step_region = self.config.train.max_iteration - self.config.train.warmup_iteration
            alpha_region = self.config.pseudo.alpha_end - self.config.pseudo.alpha_start 
            self.alpha = self.config.pseudo.alpha_start  + ((float(self.current_iteration-self.config.train.warmup_iteration)/step_region)**power)*alpha_region
    
    def adjust_beta(self, power=1.1):
        if self.current_iteration < self.offline_update_start:
            self.beta = 0.25
        else: 
            step_region = self.config.train.max_iteration - 5000
            beta_region = 0.75 - 0.25
            self.beta = 0.25  + ((float(self.current_iteration-5000)/step_region)**power)*beta_region
    
    def get_class_rate(self, logits):
        label = torch.max(logits,dim=1)[1]
        rate = np.zeros(19)
        label = label.data.cpu().numpy()
        num = label.size
        for i in range(19):
            rate[i] = len(label[label==i])/num
        return rate
    
    def sup_train_step(self, data_dict):
        self.train()
        self.optimizer_zero_grad()
        images = data_dict['image'].cuda(self.gpu_id)
        labels = data_dict['label'].cuda(self.gpu_id)
        with autocast():
            output_dict = self.seg_model(images)
            logits = output_dict['decoder_logits']
            seg_loss = self.source_seg_loss(logits, labels)

        self.scaler.scale(seg_loss).backward()
        self.scaler.step(self.seg_optimizer)
        self.scaler.update()

        self.schedule_step()

        loss_dict = {
            'seg_loss':seg_loss.clone().detach().item()
        }
        return loss_dict

    def adv_train_step(self, source_data_dict, target_data_dict):
        self.train()
        self.optimizer_zero_grad()
        source_images = source_data_dict['image'].cuda(self.gpu_id)
        source_labels = source_data_dict['label'].cuda(self.gpu_id)
        target_images = target_data_dict['image'].cuda(self.gpu_id)
        with autocast():
            source_output_dict = self.seg_model(source_images)
            source_logits = source_output_dict['decoder_logits']
            target_output_dict = self.seg_model(target_images)
            target_logits = target_output_dict['decoder_logits']
            seg_loss = self.source_seg_loss(source_logits, source_labels)
            adv_loss, dis_loss = self.get_adv_loss(source_logits=source_logits, target_logits=target_logits)
            loss = adv_loss+seg_loss
        
        self.scaler.scale(loss).backward()
        self.scaler.step(self.seg_optimizer)

        self.dis_optimizer.zero_grad()
        self.scaler.scale(dis_loss).backward()
        self.scaler.step(self.dis_optimizer)
        self.scaler.update()
        self.schedule_step()

        loss_dict = {
            'seg_loss':seg_loss.clone().detach().item(),
            'adv_loss':adv_loss.clone().detach().item(),
            'dis_loss':dis_loss.clone().detach().item()
        }

        return loss_dict

    def full2weak_logits(self, feat, params):
        tmp = []
        for i in range(feat.shape[0]):
            w, h = params['RandomScale'][0][i], params['RandomScale'][1][i]
            feat_ = F.interpolate(feat[i:i+1], size=[int(h), int(w)], mode='bilinear', align_corners=True)
            y1, y2, x1, x2 = params['RandomCrop'][0][i], params['RandomCrop'][1][i], params['RandomCrop'][2][i], params['RandomCrop'][3][i]
            y1, th, x1, tw = int(y1), int((y2-y1)), int(x1), int((x2-x1))
            feat_ = feat_[:, :, y1:y1+th, x1:x1+tw]
            if params['RandomHorizontallyFlip'][i]:
                feat_ = torch.flip(feat_, [3])
            tmp.append(feat_)
        feat = torch.cat(tmp, 0)
        feat = feat.cuda(self.gpu_id)
        return feat
    
    def full2weak_label(self, label, params):
        tmp = []
        label = label.cpu().unsqueeze(dim=1)
        label = F.interpolate(label.float(), (1024,2048), mode='nearest')
        for i in range(label.shape[0]):
            w, h = params['RandomScale'][0][i], params['RandomScale'][1][i]
            label_ = F.interpolate(label[i:i+1], size=[int(h), int(w)], mode='nearest')
            y1, y2, x1, x2 = params['RandomCrop'][0][i], params['RandomCrop'][1][i], params['RandomCrop'][2][i], params['RandomCrop'][3][i]
            y1, th, x1, tw = int(y1), int((y2-y1)), int(x1), int((x2-x1))
            label_ = label_[:, :, y1:y1+th, x1:x1+tw]
            if params['RandomHorizontallyFlip'][i]:
                label_ = torch.flip(label_, [3])
            tmp.append(label_)
        label = torch.cat(tmp, 0)
        label = label.squeeze(dim=1).cuda(self.gpu_id).long()
        return label
    
    def get_threshold(self, predicted_prob, predicted_label, use_alpha=True):
        online_pseudo_thresholds = np.zeros(19)
        valid_thresh = self.alpha
        
        if not use_alpha:
            valid_thresh = self.beta
        
        for c in range(19):
            x = predicted_prob[predicted_label==c]
            cls_num = len(x)
            if cls_num == 0:
                continue
            online_pseudo_thresholds[c] = np.percentile(x, (1-valid_thresh)*100)
        return online_pseudo_thresholds

    
    def update_offline_pseudo(self, logits, thresholds, path):
        if self.current_iteration < self.offline_update_start:
            return
        if not isinstance(logits, tuple):
            scores = F.softmax(logits, dim=1)
            full_prob_pred, full_label_pred = torch.max(scores, dim=1)
            full_prob_pred = full_prob_pred.data.cpu().numpy().copy()
            full_label_pred = full_label_pred.data.cpu().numpy().copy()
        else:
            full_label_pred, full_prob_pred = logits
        bs = full_label_pred.shape[0]
        for i in range(bs):
            label = full_label_pred[i]
            prob = full_prob_pred[i]
            label_copy = copy.deepcopy(label)
            label_cls_thresh = thresholds[label]
            ignore_index = prob < label_cls_thresh
            label[ignore_index] = 255
            old_pseudo_label = cv2.imread(path[i], -1)
            same_index = ((old_pseudo_label == label_copy) & (old_pseudo_label != 255)) 
            diff_index = ((old_pseudo_label != label_copy) & (old_pseudo_label != 255))
            label[same_index] = old_pseudo_label[same_index]
            label[diff_index] = 255
            cv2.imwrite(path[i], label)

    def generate_negative_label(self, target_mean_out, online_pseudo_label_np, online_pred_label_np):
        entropy_map_np = torch.sum(self.cal_pixel_entropy(target_mean_out), dim=1).data.cpu().numpy()
        neg_threshold1 = self.get_threshold(entropy_map_np, online_pseudo_label_np, valid_alpha=1-self.alpha)
        neg_threshold1 = neg_threshold1[online_pred_label_np]
        comp_on_label = online_pred_label_np.copy()
        neg_label = online_pred_label_np.copy()
        comp_on_label[online_pseudo_label_np!=255]=255
        neg_threshold2 = self.get_threshold(entropy_map_np, comp_on_label, valid_alpha=1-self.alpha)
        neg_threshold2 = neg_threshold2[online_pred_label_np]
        drop_cond1 = (online_pseudo_label_np!=255)&(entropy_map_np<neg_threshold1)
        drop_cond2 = (online_pseudo_label_np==255)&(entropy_map_np>neg_threshold2)
        neg_label[drop_cond1 & drop_cond2] = 255
        return neg_label

    def generate_online_pseudo(self, target_mean_out, target_label=None,target_params=None, target_path=None):
        score = F.softmax(target_mean_out, dim=1)
        
        pred_prob, pred_label = torch.max(score, dim=1)
        prob_np = pred_prob.data.cpu().numpy()
        online_label = pred_label.data.cpu().numpy()
        
        thresholds = self.get_threshold(prob_np, online_label, use_alpha=True)
        label_cls_thresh = thresholds[online_label]
        ignore_index = prob_np < label_cls_thresh
        online_label[ignore_index] = 255

        online_label = torch.from_numpy(online_label).cuda(self.gpu_id).long()

        return pred_label, online_label

    def self_train_step(self, source_data_dict, target_data_dict):
        self.train()
        self.optimizer_zero_grad()
        source_images = source_data_dict['image'].cuda(self.gpu_id)
        source_labels = source_data_dict['label'].cuda(self.gpu_id)
        with autocast():
            source_output_dict = self.seg_model(source_images)
            source_logits = source_output_dict['decoder_logits']
            s_seg_loss = self.seg_loss(source_logits, source_labels)

        self.scaler.scale(s_seg_loss).backward()
        target_images = target_data_dict['image'].cuda(self.gpu_id)
        target_full_images = target_data_dict['full_image'].cuda(self.gpu_id)
        target_labels = target_data_dict['label'].cuda(self.gpu_id)

        loss = torch.tensor(0.0).to(self.gpu_id)
        with autocast():
            target_output_dict = self.seg_model(target_images)
            target_logits = target_output_dict['decoder_logits']
            # t_seg_loss = self.seg_loss(target_logits, target_labels)
            t_seg_loss = self.target_seg_loss(target_logits, target_labels)
            loss += t_seg_loss
            # offline_ent_reg, offline_kld_reg = self.get_iast_regular(target_logits, target_labels)
            consist_loss = self.get_consistancy_loss(target_logits, target_images)
            consist_loss = self.get_consistancy_loss(target_logits, target_full_images, target_params=target_data_dict['param'])
            loss += consist_loss
            # online_pred_label, online_pseudo_label = self.get_online_pseudo_label(target_data=target_images, target_label=target_labels)
            online_pred_label, online_pseudo_label = self.get_online_pseudo_label(target_data=target_full_images,
                                                                                  target_label=target_labels.detach(),
                                                                                  target_params=target_data_dict['param'],
                                                                                  target_path=target_data_dict['name'])

            online_pseudo_loss = self.seg_loss(target_logits, online_pseudo_label)*0.5
            loss += online_pseudo_loss
            online_ent_reg, online_kld_reg = self.get_iast_regular(target_logits, online_pseudo_label)

            # ent_reg_loss = offline_ent_reg+online_ent_reg*0.5
            ent_reg_loss = online_ent_reg*0.5
            # kld_reg_loss = offline_kld_reg+online_kld_reg*0.5
            kld_reg_loss = online_kld_reg*0.5
            loss += ent_reg_loss
            loss += kld_reg_loss

        self.scaler.scale(loss).backward()
        self.scaler.step(self.seg_optimizer)
        self.scaler.update()
        self.schedule_step()
 
        self.mean_model.update_parameters(self.seg_model)

        loss_dict = {'pseudo_seg_loss': t_seg_loss.clone().detach().item(),}

        if source_images is not None:
            loss_dict.update({'source_seg_loss': s_seg_loss.clone().detach().item()})

        if self.mean_model is not None:
            loss_dict.update({'consist_loss': consist_loss.clone().detach().item(),
                              'online_loss': online_pseudo_loss.clone().detach().item(),
                              'ent_reg_loss': ent_reg_loss.clone().detach().item(),
                              'kld_reg_loss': kld_reg_loss.clone().detach().item(),
                              })
        self.current_iteration += 1
        self.adjust_alpha()
        self.adjust_beta()
        return loss_dict
    

    def distillation_step(self, source_data_dict, target_data_dict):
        # print(target_data_dict['name'])
        self.train()
        self.optimizer_zero_grad()
        self.mean_model.eval()
        source_images = source_data_dict['image'].cuda(self.gpu_id)
        source_labels = source_data_dict['label'].cuda(self.gpu_id)
        with autocast():
            source_output_dict = self.seg_model(source_images)
            source_logits = source_output_dict['decoder_logits']
            s_seg_loss = self.seg_loss(source_logits, source_labels)
        self.scaler.scale(s_seg_loss).backward()

        target_images = target_data_dict['image'].cuda(self.gpu_id)
        target_labels = target_data_dict['label'].cuda(self.gpu_id)
        # target_full_images = target_data_dict['full_image'].cuda(self.gpu_id)
        loss = torch.tensor(0.0).to(self.gpu_id)
        with autocast():
            target_output_dict = self.seg_model(target_images)
            target_logits = target_output_dict['decoder_logits']
            t_seg_loss = self.seg_loss(target_logits, target_labels)
            loss += t_seg_loss
            student = F.softmax(target_logits, dim=1)

            with torch.no_grad():
                teacher_output_dict = self.mean_model(target_images)
                teacher_logits = teacher_output_dict['decoder_logits']
                teacher = F.softmax(teacher_logits, dim=1).detach()

            distill_loss = F.kl_div(student, teacher, reduce='mean')
            loss += 1*distill_loss
        
        self.scaler.scale(loss).backward()
        self.scaler.step(self.seg_optimizer)
        self.scaler.update()
        self.schedule_step()

        loss_dict = {
            'pseudo_seg_loss': t_seg_loss.clone().detach().item(),
            'distill_loss': distill_loss.clone().detach().item(),
        }
        if source_images is not None:
            loss_dict.update({'source_seg_loss': s_seg_loss.clone().detach().item()})
        return loss_dict

        
    def save_logits(self, logits, prefix):
        probs = F.softmax(logits, dim=1)
        labels = torch.argmax(probs, dim=1)
        labels = labels.data.cpu().numpy()
        for i in range(labels.shape[0]):
            label_i = labels[i]
            cv2.imwrite('./vis/'+prefix+'_bs_'+str(i)+'.png', label_i)

    def save_labels(self, labels, prefix):
        labels = labels.data.cpu().numpy()
        for i in range(labels.shape[0]):
            label_i = labels[i]
            cv2.imwrite('./vis/'+prefix+'_bs_'+str(i)+'.png', label_i)
    
    def save_image(self, image, prefix):
        image = image.data.cpu().numpy()
        image *= self.config.input.std
        image += self.config.input.mean
        for i in range(image.shape[0]):
            image_i = np.transpose(image[i], (1,2,0))
            cv2.imwrite('./vis/'+prefix+'_bs_'+str(i)+'.png', image_i)

    def get_adv_loss(self, source_logits, target_logits):
        s_D_logits = self.discriminator(F.softmax(source_logits, dim=1).detach())
        t_D_logits = self.discriminator(F.softmax(target_logits, dim=1).detach())

        is_source = torch.zeros_like(s_D_logits).cuda(self.gpu_id)
        is_target = torch.ones_like(t_D_logits).cuda(self.gpu_id)

        dis_loss = (self.adv_loss(s_D_logits, is_source) +
                              self.adv_loss(t_D_logits, is_target)) / 2

        t_D_logits = self.discriminator(F.softmax(target_logits, dim=1))
        adv_loss = self.adv_loss(t_D_logits, is_source)*self.adv_loss_weight

        return adv_loss, dis_loss

    def get_iast_regular(self, target_logits, target_label=None):
        weight1 = (target_label==255).float().detach()
        weight2 = (target_label!=255).float().detach()

        ent_regular_loss = self.ent_reg_loss(target_logits, weight1) * 2
        kld_regular_loss = self.kld_reg_loss(target_logits, weight2) * 0.1
        return ent_regular_loss, kld_regular_loss

    def cal_pixel_entropy(self, logits):
        score = F.softmax(logits, dim=1)
        logits_log_softmax = torch.log(score+1e-8)
        entropy = -score*logits_log_softmax
        return entropy

    def ent_reg_loss(self, logits, weight=None):
        num_class = logits.shape[1]
        val_num = weight[weight > 0].numel()*num_class
        entropy = self.cal_pixel_entropy(logits)
        if val_num > 0:
            entropy_reg = torch.sum(torch.sum(entropy, dim=1)*weight) / val_num
        else:
            entropy_reg = torch.tensor(0.0).to(self.gpu_id)
        return entropy_reg

    def kld_reg_loss(self, logits, weight=None):
        assert weight is not None
        num_class=logits.shape[1]
        val_num = weight[weight > 0].numel()*num_class
        logits_log_softmax = torch.log_softmax(logits, dim=1)
        num_classes = logits.size()[1]
        kld = (-1.0/num_classes) * logits_log_softmax
        kld_reg = torch.sum(torch.sum(kld,dim=1)*weight) / val_num
        return kld_reg

    def get_class_balance_weight(self, logits):
        label = torch.max(logits,dim=1)[1]
        rate = np.zeros(19)
        label = label.data.cpu().numpy()
        num = label.size
        for i in range(19):
            rate[i] = len(label[label==i])/num
        weight = 0.8 + np.exp(-10*rate)
        weight_map = label[weight]
        weight_map = torch.from_numpy(weight_map).float().cuda(self.gpu_id)
        return weight_map

    def tsallis_entropy(self, scores):
        v = 1.5
        scale = 1.0/(v-1.0)
        tsallis_entropy = 1-torch.sum(torch.pow(scores, v), dim=1)
        tsallis_entropy = scale*tsallis_entropy
        return tsallis_entropy

    def weighted_max_square_loss(self, pred_probs, pred_labels):
        pred_labels_np = pred_labels.data.cpu().numpy()
        label_count = np.bincount(pred_labels_np)
        weight_map = label_count[pred_labels]
        bs, num_class, h, w = pred_probs.shape
        N = bs*h*w
        weight_map = 1/(2*np.power(weight_map, 0.2)*(N**0.8))
        weight_map = torch.from_numpy(weight_map).cuda(self.gpu_id).float()
        weight_map = weight_map.unsqueeze(dim=1)
        loss = -torch.sum(torch.pow(pred_probs, 2)*weight_map)
        return loss

    def max_square_loss(self, pred_prob, labels):
        loss = -torch.mean(torch.sum(torch.pow(pred_prob, 2),dim=1))/2
        return loss

    def get_max_square_loss(self, logits, labels, T=1):
        scores = F.softmax(logits/T,dim=1)
        return self.max_square_loss(scores, labels)*0.3

    def get_negative_label(self, pred_label):
        neg_label = pred_label.copy()
        for lab in range(19):
            tmpLab = rd.randint(0,18)
            while tmpLab == lab:
                tmpLab = rd.randint(0,18)
            neg_label[pred_label == lab] = tmpLab
        neg_label = torch.from_numpy(neg_label).cuda(self.gpu_id).long()
        return neg_label

    def negative_loss(self, scores, pred_labels, labels, ignore_index=255):
        pred_labels_np = pred_labels.data.cpu().numpy()
        labels_np = labels.data.cpu().numpy()
        mask = labels_np != ignore_index
        mask = torch.from_numpy(mask).cuda(self.gpu_id).float()
        neg_label = self.get_negative_label(pred_labels_np)
        neg_label_onehot = F.one_hot(neg_label, 19).permute(0,3,1,2).detach()
        weight_map = torch.max(scores*neg_label_onehot, dim=1)[0].detach()
        loss = -torch.sum(mask*weight_map*torch.sum(torch.log(1-scores+1e-10)*neg_label_onehot, dim=1))/(mask.sum())
        return loss

    def get_negative_loss(self, logits, online_pred_labels, neg_labels, T=1.0):
        scores = F.softmax(logits/T, dim=1)
        return self.negative_loss(scores, online_pred_labels.detach(), neg_labels.detach())

    def generate_pseudo_label(self, pseudo_dataloader):
        print("generate pseudo label")
        self.pseudo_selector.solve(model=self.seg_model, pseudo_dataloader=pseudo_dataloader, gpu_id=self.gpu_id)
        print("finish generate pseudo label")
    
    def evaluate(self, train_dataloader, eval_dataloader):
        self.seg_model.eval()
        if self.config.use_ema:
            if self.mean_model is not None:
                self.mean_model.eval()
                self.mean_update_bn(train_dataloader, self.mean_model)
        n_class = self.config.dataset.num_classes
        intersection_sum = 0
        union_sum = 0

        with torch.no_grad():
            for i, eval_data_dict in enumerate(eval_dataloader):
                images = eval_data_dict['image']; labels = eval_data_dict['label']
                images, labels = images.cuda(self.gpu_id), labels.cuda(self.gpu_id)
                if self.config.use_ema:
                    logits = self.mean_model(images)['decoder_logits']
                else:
                    logits = self.seg_model(images)['decoder_logits']
                label_pred = logits.max(dim=1)[1]
                intersection, union = intersectionAndUnionGPU(label_pred, labels, n_class)
                intersection_sum += intersection
                union_sum += union
        if self.config.distribution.num_gpus > 1:
            dist.all_reduce(intersection_sum), dist.all_reduce(union_sum)
        intersection_sum = intersection_sum.cpu().numpy()
        union_sum = union_sum.cpu().numpy()
        iu = intersection_sum / (union_sum + 1e-10)
        mean_iu = np.mean(iu)
        torch.cuda.empty_cache()
        return iu, mean_iu

    def get_state_dict(self):
        state_dict = {
            'seg_model': self.seg_model.state_dict()
        }
        if self.discriminator is not None:
            state_dict.update({
                'dis_model': self.discriminator.state_dict()
            })
        if self.mean_model is not None:
            state_dict.update({'mean_model': self.mean_model.state_dict()})
        return state_dict

    def load_state_dict(self, state_dict):
        self.seg_model.load_state_dict(state_dict['seg_model'])
        if self.discriminator is not None:
            self.discriminator.load_state_dict(state_dict['dis_model'])
        if self.mean_model is not None:
            self.mean_model.load_state_dict(state_dict['mean_model'])

    @torch.no_grad()
    def mean_update_bn(self, loader, model):
        momenta = {}
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.running_mean = torch.zeros_like(module.running_mean)
                module.running_var = torch.ones_like(module.running_var)
                momenta[module] = module.momentum
        if not momenta:
            return
        was_training = model.training
        model.train()
        for module in momenta.keys():
            module.momentum = None
            module.num_batches_tracked *= 0

        for input_dict in loader:
            input = input_dict['image'].cuda(self.gpu_id)
            model(input)

        for bn_module in momenta.keys():
            bn_module.momentum = momenta[bn_module]
        model.train(was_training)
        torch.cuda.empty_cache()

    def get_consistancy_loss(self, logits, images, target_params=None):
        with torch.no_grad():
            mean_logits = self.mean_model(images)['decoder_logits'].detach()
            if target_params is not None:
                mean_logits = self.full2weak_logits(mean_logits, target_params).detach()
        mean_consis = self.consist_loss(F.softmax(logits, dim=1), F.softmax(mean_logits, dim=1))
        return mean_consis

    def get_online_pseudo_label(self, target_data, target_label=None, target_params=None, target_path=None):
        self.mean_model.eval()
        with torch.no_grad():
            mean_pred = self.mean_model(target_data)['decoder_logits'].detach()
            
            if target_path is not None:
                scores = F.softmax(mean_pred.detach(), dim=1).detach()
                prob, label = torch.max(scores, dim=1)
                offline_update_threshold = self.get_threshold(prob.data.cpu().numpy(), label.data.cpu().numpy(), use_alpha=False)
                offline_update_threshold[offline_update_threshold>0.99]=0.99
                self.update_offline_pseudo(mean_pred, offline_update_threshold, target_path)
            
            if target_params is not None:
                mean_pred = self.full2weak_logits(mean_pred, target_params).detach()
        online_pred_label, online_pseudo_label = self.generate_online_pseudo(mean_pred, target_label, target_params, target_path)
        online_pred_label, online_pseudo_label = online_pred_label.detach(), online_pseudo_label.detach()
        return online_pred_label, online_pseudo_label

    def update_pseudo_parameter(self):
        return 0