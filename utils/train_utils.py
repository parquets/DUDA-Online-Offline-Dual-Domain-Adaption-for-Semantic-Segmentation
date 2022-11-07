import os
import random
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
from dataset import Synthia, Cityscapes, GTAV
from torch.utils.data import  DataLoader
from models.schedulers import CosineAnnealingLR_with_Restart, PolynomialLR, WarmUpLR



def build_optimizer(config, model, model_type="seg_model"):
    optim_type = config.train.optimizer
    if model_type == "seg_model":
        params = model.get_params(config.train)
        if (optim_type == 'SGD'):
            optimizer = optim.SGD(params, momentum=0.9, nesterov=True)
        elif (optim_type == 'Adam'):
            optimizer = optim.Adam(params, betas=(0.9, 0.999))
        else:
            optimizer = optim.SGD(params, momentum=0.9, nesterov=True)
        return optimizer
    elif model_type == "dis_model":
        if isinstance(model, list):
            dis_params = []
            for m in model:
                dis_params.append({'params': m.parameters()})
        else:
            dis_params = model.parameters()
        optimizer = optim.Adam(dis_params, lr=config.train.discriminator_lr, betas=(0.9, 0.999))
        return optimizer
    else:
        raise "Unrecognized model type"

def build_scheduler(config, optimizer):
    if ("Cos" in config.train.lr_scheduler) or ("cos" in config.train.lr_scheduler):
        seg_scheduler = CosineAnnealingLR_with_Restart(optimizer, 
                                                      T_max=config.train.cos_scheduler_tmax*config.train.max_iteration//config.train.max_epoch,
                                                      T_mult=config.train.cos_scheduler_tmult,
                                                      eta_min=config.train.lr*0.001)
    elif ("Poly" in config.train.lr_scheduler) or ("poly" in config.train.lr_scheduler):
        seg_scheduler = PolynomialLR(optimizer, max_iter=config.train.max_iteration, gamma=config.train.scheduler_gamma)
    else:
        raise "Unrecognized schedule"
    if config.train.warmup_iteration > 0:
        seg_scheduler = WarmUpLR(optimizer, seg_scheduler, warmup_iters=config.train.warmup_iteration)
    return seg_scheduler
