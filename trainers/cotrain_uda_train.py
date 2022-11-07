import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast as autocast
import torch.optim.swa_utils

class CoUDATrainer:
    def __init__(self, config, scaler, seg_model_A, seg_model_B, 
                 seg_optimizer_A, seg_optimizer_B, 
                 seg_scheduler_A, seg_scheduler_B,
                 pseudo_selector=None, gpu_id=0) -> None:
        self.config = config
        self.gpu_id = gpu_id
        self.rank = config.distribution.rank_within_nodes*config.distribution.num_gpus + gpu_id
        self.num_classes = config.dataset.num_classes
        self.seg_model_A = seg_model_A
        self.seg_model_B = seg_model_B
        self.mean_model_A = torch.optim.swa_utils.AveragedModel(self.seg_model_A, device=gpu_id)
        self.mean_model_B = torch.optim.swa_utils.AveragedModel(self.seg_model_B, device=gpu_id)
        self.mean_model_A.update_parameters(self.seg_model_A)
        self.mean_model_B.update_parameters(self.seg_model_B)
        self.seg_optimizer_A = seg_optimizer_A
        self.seg_optimizer_B = seg_optimizer_B
        self.seg_scheduler_A = seg_scheduler_A
        self.seg_scheduler_B = seg_scheduler_B
        self.pseudo_selector = pseudo_selector
        self.scaler = scaler
        self.source_seg_loss = nn.CrossEntropyLoss(ignore_index=255).cuda(gpu_id)
        self.target_seg_loss = nn.CrossEntropyLoss(ignore_index=255).cuda(gpu_id)

    def get_lr(self):
        lr = self.seg_optimizer.param_groups[-1]['lr']
        return lr

    def train(self):
        self.seg_model_A.train()
        self.seg_model_B.train()

    def eval(self):
        self.seg_model_A.eval()
        self.seg_model_B.eval()

    def optimizer_zero_grad(self):
        self.seg_optimizer_A.zero_grad()
        self.seg_optimizer_B.zero_grad()

    def optimizer_step(self):
        self.seg_optimizer_A.step()
        self.seg_optimizer_B.step()
    
    def schedule_step(self):
        self.seg_scheduler_A.step()
        self.seg_scheduler_B.step()

    def convert_sync_batchnorm(self):
        self.seg_model_A = nn.SyncBatchNorm.convert_sync_batchnorm(self.seg_model_A)
        self.seg_model_B = nn.SyncBatchNorm.convert_sync_batchnorm(self.seg_model_B)
    
    def model_ddp(self):
        self.seg_model_A = nn.parallel.DistributedDataParallel(self.seg_model_A, device_ids=[self.gpu_id], find_unused_parameters=True)
        self.seg_model_B = nn.parallel.DistributedDataParallel(self.seg_model_B, device_ids=[self.gpu_id], find_unused_parameters=True)

    def co_train_step(self, source_data, source_label, target_data, target_label):
        self.train()
        self.optimizer_zero_grad()
        if (source_data is not None) and (source_label is not None):
            with autocast():
                source_output_A = self.seg_model_A(source_data)
                source_output_B = self.seg_model_B(source_data)
                s_seg_loss_A = self.source_seg_loss(source_output_A, source_label)
                s_seg_loss_B = self.source_seg_loss(source_output_B, source_label)
                s_seg_loss = s_seg_loss_A+s_seg_loss_B
            self.scaler.scale(s_seg_loss).backward()

        loss_A = torch.tensor(0.0).to(self.gpu)
        loss_B = torch.tensor(0.0).to(self.gpu)
        with autocast():
            target_output_A = self.seg_model_A(target_data)
            target_output_B = self.seg_model_B(target_data)
            loss_A += self.target_seg_loss(target_output_A, target_label)
            loss_B += self.target_seg_loss(target_output_B, target_label)
            consist_loss_A, consist_loss_B = self.temporal_consist_loss(target_data, target_output_A, target_output_B)   
            loss_A += consist_loss_A
            loss_B += consist_loss_B 

        self.scaler.scale(loss_A).backward()
        self.scaler.scale(loss_B).backward()
        self.scaler.step(self.seg_optimizer_A)
        self.scaler.step(self.seg_optimizer_B)
        self.scaler.update()
        self.schedule_step()
        self.mean_model_A.update_parameters(self.seg_model_A)
        self.mean_model_B.update_parameters(self.seg_model_B)

        loss_dict = {}
        return loss_dict

    def temporal_consist_loss(self, target_data, pred_A, pred_B):
        with torch.no_grad():
            ema_pred_A = self.mean_model_A(target_data).detach()
            ema_pred_B = self.mean_model_B(target_data).detach()
        consist_loss_A = F.mse_loss(F.softmax(pred_A, dim=1), F.softmax(ema_pred_A, dim=1))
        consist_loss_B = F.mse_loss(F.softmax(pred_B, dim=1), F.softmax(ema_pred_B, dim=1))
        return consist_loss_A, consist_loss_B

    def online_pseudo_loss(self, target_data, pred_A, pred_B):
        self.mean_model_A.eval()
        self.mean_model_B.eval()

    def get_small_loss(self, loss_map):
        return 0

