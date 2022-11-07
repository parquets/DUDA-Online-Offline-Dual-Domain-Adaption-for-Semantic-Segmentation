import os
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast as autocast
import torch.distributed as dist
from utils.metric import intersectionAndUnionGPU

class Trainer(object):
    def __init__(self, config, scaler, train_model, train_optimizer, train_scheduler, train_loss, gpu_id) -> None:
        super().__init__()
        self.config = config
        self.train_model = train_model
        self.train_optimizer = train_optimizer
        self.train_loss = train_loss
        self.train_scheduler = train_scheduler
        self.gpu_id = gpu_id
        self.scaler = scaler

    def get_lr(self):
        lr = self.seg_optimizer.param_groups[-1]['lr']
        return lr

    def train(self):
        self.train_model.train()

    def eval(self):
        self.train_model.eval()

    def optimizer_zero_grad(self):
        self.train_optimizer.zero_grad()

    def optimizer_step(self):
        self.train_optimizer.step()
    
    def schedule_step(self):
        self.train_scheduler.step()

    def convert_sync_batchnorm(self):
        self.train_model = nn.SyncBatchNorm.convert_sync_batchnorm(self.train_model)
    
    def model_ddp(self):
        self.train_model = nn.parallel.DistributedDataParallel(self.train_model, device_ids=[self.gpu_id], find_unused_parameters=True)
    
    def sup_train_step(self, images, labels):
        self.train()
        self.optimizer_zero_grad()
        with autocast():
            output_dict = self.train_model(images)
            logits = output_dict['decoder_logits']
            loss = self.train_loss(logits, labels)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.seg_optimizer)
        self.scaler.update()

        self.schedule_step()

        loss_dict = {
            'seg_loss':loss.clone().detach().item()
        }
        return loss_dict

    def evaluate(self, eval_dataloader):
        self.eval()
        n_class = self.config.dataset.num_classes
        intersection_sum = 0
        union_sum = 0
        with torch.no_grad():
            for i, eval_data_dict in enumerate(eval_dataloader):
                images = eval_data_dict['image']; labels = eval_data_dict['label']
                images, labels = images.cuda(self.gpu_id), labels.cuda(self.gpu_id)
                logits = self.train_model(images)['decoder_logits']
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
        return iu, mean_iu