import os
from models import GeneralSegmentor
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
import time
import random
import pprint
import torch
import torch.nn as nn
import numpy as np
import torch.multiprocessing as mp
import torch.distributed as dist
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
from utils.logger import create_logger
logger = create_logger(config, config.logger.train_log_save_dir)

'''
python train.py --cfg ./configs/syn2cityscapes/source_only_res101.yaml
nohup python -u train.py --cfg ./configs/syn2cityscapes/source_only_res101.yaml >>./logs/source_only.out
nohup python -u train.py --cfg ./configs/syn2cityscapes/source_only_res101_multi.yaml >>./logs/source_only_slpit.out

python train.py --cfg ./configs/syn2cityscapes/adv_train_res101.yaml
nohup python -u train.py --cfg ./configs/syn2cityscapes/adv_train_res101.yaml >>./logs/adv_train_0110.out

python train.py --cfg ./configs/syn2cityscapes/IAST/self_train_res101_update.yaml
nohup python -u train.py --cfg ./configs/syn2cityscapes/IAST/self_train_res101.yaml >>./logs/st_0121.out
'''

result = []

def seed_everything(seed=888):
   random.seed(seed)
   os.environ['PYTHONHASHSEED'] = str(seed)
   np.random.seed(seed)
   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   torch.backends.cudnn.benchmark = False
   torch.backends.cudnn.deterministic = True


def train_model(config, trainer, source_train_loader, target_train_loader, target_val_loader, gpu_id):
    rank = config.distribution.rank_within_nodes*config.distribution.num_gpus + gpu_id
    max_iteration = config.train.max_iteration
    display_iteration = config.train.display_iteration
    eval_iteration = config.train.eval_iteration
    
    curr_iter = 1
    curr_epoch = 1
    log_total_loss = {}
    log_total_loss['loss'] = 0
    best_iou = 0

    source_iter = iter(source_train_loader)
    target_iter = iter(target_train_loader)

    iter_report_start = time.time()
    while curr_iter <= max_iteration:
        try:
            source_data_dict = next(source_iter)
        except StopIteration:
            source_iter = iter(source_train_loader)
            source_data_dict = next(source_iter)
        try:
            target_data_dict = next(target_iter)
        except StopIteration:
            target_iter = iter(target_train_loader)
            target_data_dict = next(target_iter)
            curr_epoch += 1

        if config.stage == "source_only":
            loss_dict = trainer.sup_train_step(source_data_dict)
        elif config.stage == "adv_train":
            loss_dict = trainer.adv_train_step(source_data_dict, target_data_dict)
        elif config.stage == "self_train":
            loss_dict = trainer.self_train_step(source_data_dict, target_data_dict)
        elif config.stage == "distill":
            loss_dict = trainer.distillation_step(source_data_dict, target_data_dict)
        else:
            raise "Unrecognized stage!"
        
        for loss_name, loss_item in loss_dict.items():
            log_total_loss[loss_name] = loss_item if loss_name not in log_total_loss else log_total_loss[loss_name] + loss_item
            log_total_loss['loss'] += loss_item

        if curr_iter % display_iteration == 0 and rank==0:
            iter_report_end = time.time()
            iter_report_time = iter_report_end-iter_report_start
            eta = itv2time(iter_report_time * (max_iteration - curr_iter) / display_iteration)
            report = 'eta: {}, iter: {}, time: {:.3f} s/iter, lr: {:.2e}'.format(eta, curr_iter, iter_report_time/display_iteration, trainer.get_lr()) + print_loss_dict(log_total_loss, config.train.display_iteration)
            logger.info(report)
            log_total_loss = {}
            log_total_loss['loss'] = 0
            iter_report_start = time.time()
            
        if curr_iter % eval_iteration == 0 or curr_iter == 600: # and curr_iter >= 10000:
            iu, miou = trainer.evaluate(target_train_loader, target_val_loader)
            if rank == 0:
                result_item = {'epoch': curr_iter}
                result_item.update({'iou': miou})
                result_item.update(result_list2dict(iu,'iou'))
                result.append(result_item)
                report = 'iter {}, val_miou: {:.6f}({:.6f})'.format(curr_iter, miou, print_top(result, 'iou')) + print_iou_list(iu)
                logger.info(report)
                torch.save(trainer.mean_model.state_dict(), config.save.save_dir+'iter_'+str(curr_iter)+'mean_model.pth')
                torch.save(trainer.seg_model.state_dict(), config.save.save_dir+'iter_'+str(curr_iter)+'seg_model.pth')
                if miou > best_iou:
                    torch.save(trainer.mean_model.state_dict(), config.save.save_dir+'best_mean_model.pth')
                    best_iou = miou
        curr_iter += 1

    iu, miou = trainer.evaluate(target_train_loader, target_val_loader)
    if rank == 0:
        torch.save(trainer.mean_model.state_dict(), config.save.save_dir+'last_iter_mean.pth')
        torch.save(trainer.seg_model.state_dict(), config.save.save_dir+'last_iter_seg.pth')
        logger.info("final miou:"+str(float(miou)))
        logger.info("finish training")


def main(gpu_id, config):
    seed_everything()
    multi_gpu = config.distribution.num_gpus > 1
    rank = config.distribution.rank_within_nodes*config.distribution.num_gpus + gpu_id
    if multi_gpu:
        torch.cuda.set_device(gpu_id)
        dist.init_process_group(backend='nccl', 
                                init_method='env://', 
                                world_size=config.distribution.world_size, 
                                rank=rank)
    num_classes = config.dataset.num_classes
    seg_model = GeneralSegmentor(config, num_classes)
    seg_optimizer = build_optimizer(config, seg_model)
    seg_scheduler = build_scheduler(config, seg_optimizer)
    source_seg_loss, target_seg_loss = build_loss(config)
    pseudo_selector = IAST(config)
    # test_seg_scheduler(seg_optimizer, seg_scheduler)
    scaler = GradScaler()
    trainer = UDATrainer(config,
                         scaler=scaler,
                         seg_model=seg_model,
                         seg_optimizer=seg_optimizer,
                         source_seg_loss=source_seg_loss,
                         target_seg_loss=target_seg_loss,
                         seg_scheduler=seg_scheduler,
                         pseudo_selector=pseudo_selector,
                         gpu_id=gpu_id)
    if config.distribution.num_gpus > 1:
        trainer.convert_sync_batchnorm()
        trainer.model_ddp()

    source_train_loader, target_train_loader, target_val_loader = build_uda_dataloader(config, multi_gpu, rank)
    train_model(config, trainer, source_train_loader, target_train_loader, target_val_loader, gpu_id)
    if multi_gpu:
        dist.destroy_process_group()

if __name__ == '__main__':
    IMAGE_NUM = config.source_dataset.image_num
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = config.distribution.port
    config.distribution.world_size = config.distribution.num_gpus*config.distribution.num_nodes
    if config.train.max_epoch != -1:
        config.train.max_iteration = (IMAGE_NUM*config.train.max_epoch) // (config.train.source_bs*config.distribution.num_gpus)
    logger.info('training config:{}\n'.format(pprint.pformat(config)))
    if config.distribution.num_gpus > 1:
        mp.spawn(main, nprocs=config.distribution.num_gpus, args=(config, ))
    else:
        main(0, config)