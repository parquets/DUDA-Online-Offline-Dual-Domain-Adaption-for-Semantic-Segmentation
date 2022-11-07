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
from models.deeplabv2 import Deeplab
from models import GeneralSegmentor
import copy
from PIL import Image
import time
import random as rd
import math


def bgr2rbg(_data):
    if isinstance(_data, torch.Tensor):
        _data = _data.data.cpu().numpy()
    _data = _data[:,::-1,:,:]
    _data = torch.from_numpy(_data.copy()).float()
    return _data

class DistillTrainer:
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
        self.seg_model

        if config.train.resume_from:
            self.seg_model.cuda(gpu_id)
            self.seg_model.load_pretrained(config.train.resume_from, gpu_id)

        self.mean_model = None
        if config.use_ema:
            self.mean_model = torch.optim.swa_utils.AveragedModel(self.seg_model, device=gpu_id)
            self.mean_model.update_parameters(self.seg_model)
            assert self.is_equal(self.seg_model, self.mean_model)

        if config.use_distill:
            # self.mean_model = Deeplab(is_teacher=False, bn_clr=True)
            self.mean_model = GeneralSegmentor(config, config.dataset.num_classes)
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

    def distillation_step(self, source_data_dict, target_data_dict):
        # print(target_data_dict['name'])
        self.train()
        self.optimizer_zero_grad()
        self.mean_model.eval()
        source_images = source_data_dict['image']
        source_images = bgr2rbg(source_images).cuda(self.gpu_id)
        source_labels = source_data_dict['label'].cuda(self.gpu_id)
        with autocast():
            source_output_dict = self.seg_model(source_images/255)
            source_logits = source_output_dict['decoder_logits']
            s_seg_loss = self.seg_loss(source_logits, source_labels)
        self.scaler.scale(s_seg_loss).backward()

        target_images = target_data_dict['image']
        rbg_target_images = bgr2rbg(target_images)
        target_images = target_images.cuda(self.gpu_id)
        rbg_target_images = rbg_target_images.cuda(self.gpu_id)

        target_labels = target_data_dict['label'].cuda(self.gpu_id)
        # target_full_images = target_data_dict['full_image'].cuda(self.gpu_id)
        loss = torch.tensor(0.0).to(self.gpu_id)
        with autocast():
            target_output_dict = self.seg_model(rbg_target_images/255)
            target_logits = target_output_dict['decoder_logits']
            # t_seg_loss = self.seg_loss(target_logits, target_labels)
            t_seg_loss = self.target_seg_loss(target_logits, target_labels)
            loss += t_seg_loss
            student = F.softmax(target_logits, dim=1)

            with torch.no_grad():
                teacher_output_dict = self.mean_model(target_images)
                teacher_logits = teacher_output_dict['decoder_logits']
                teacher = F.softmax(teacher_logits, dim=1).detach()

            distill_loss = F.kl_div(student, teacher, reduce='mean')
            loss += distill_loss
        
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
                # images, labels = images.cuda(self.gpu_id), labels.cuda(self.gpu_id)
                labels = labels.cuda(self.gpu_id)
                images = bgr2rbg(images).cuda(self.gpu_id)
                if self.config.use_ema:
                    logits = self.mean_model(images)['decoder_logits']
                else:
                    logits = self.seg_model(images/255)['decoder_logits']
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

    def evaluate_teacher(self, train_dataloader, eval_dataloader):
        print("evaluate_teacher")
        self.mean_model.eval()
        n_class = self.config.dataset.num_classes
        intersection_sum = 0
        union_sum = 0
        with torch.no_grad():
            for i, eval_data_dict in enumerate(eval_dataloader):
                images = eval_data_dict['image']; labels = eval_data_dict['label']
                # images = bgr2rbg(images)
                images, labels = images.cuda(self.gpu_id), labels.cuda(self.gpu_id)
            
                logits = self.mean_model(images)['decoder_logits']
    
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
