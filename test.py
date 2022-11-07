import imp
import os
from models import GeneralSegmentor
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import time
import random
import pprint
import torch
import torch.nn as nn
import numpy as np
import torch.multiprocessing as mp
import torch.distributed as dist
import cv2
from utils.train_utils import build_optimizer, build_scheduler
from torch.cuda.amp import autocast as autocast, GradScaler
from trainers.uda_trainer import UDATrainer
from utils.display_utils import result_list2dict, print_top, print_iou_list, itv2time, print_loss_dict
from dataset import build_uda_dataloader
from models import build_loss
from pseudo.generate_pseudo_label_iast import IAST
from configs.config import config
from configs.parse_args import parse_args
args = parse_args()


def seed_everything(seed=888):
   random.seed(seed)
   os.environ['PYTHONHASHSEED'] = str(seed)
   np.random.seed(seed)
   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   torch.backends.cudnn.benchmark = False
   torch.backends.cudnn.deterministic = True


def norm_reverse(data, mean=[104.00698793, 116.66876762, 122.67891434], std=[1,1,1]):
    return (data+np.array(mean))/np.array(std)

def main():
    print("main")
    source_train_loader, target_train_loader, target_val_loader = build_uda_dataloader(config, False, 0)
    
    source_iter = iter(source_train_loader)
    target_iter = iter(target_train_loader)

    source_data_dict = next(source_iter)
    target_data_dict = next(target_iter)

    target_images = target_data_dict['image'].data.cpu().numpy()
    target_full_images = target_data_dict['full_image'].data.cpu().numpy()
    target_labels = target_data_dict['label'].data.cpu().numpy()

    target_images = np.transpose(target_images, (0,2,3,1))
    target_full_images = np.transpose(target_full_images, (0,2,3,1))
    target_images = norm_reverse(target_images)
    target_full_images = norm_reverse(target_full_images)

    batch_size = target_images.shape[0]
    for bs in range(batch_size):
        image = target_images[bs]
        full_image = target_full_images[bs]

        cv2.imwrite('./vis/image_'+str(bs)+'.png', image)
        cv2.imwrite('./vis/full_image_'+str(bs)+'.png', full_image)





if __name__ == '__main__':
    seed_everything()
    config.target_loader.bs=32
    main()


# python test.py --cfg ./configs/gtav2cityscapes/IAST/self_train_res101_update.yaml