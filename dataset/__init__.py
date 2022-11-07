from torch.utils.data import DataLoader
from .synthia import Synthia
from .cityscapes import Cityscapes
from .gtav import GTAV

from torch.utils.data.distributed import DistributedSampler


def build_uda_dataloader(config, multi_gpu=False, rank=0):
    source_train_set = None
    if config.source_dataset.dataset_type == "Synthia":
        source_train_set = Synthia(config, 'train', 'source')
    elif config.source_dataset.dataset_type == "GTAV":
        source_train_set = GTAV(config, 'train', 'source')
    else:
        raise "Not implement dataset"
    if multi_gpu:
        source_train_sampler = DistributedSampler(source_train_set, num_replicas=config.distribution.world_size, rank=rank)
        source_train_loader = DataLoader(dataset=source_train_set, batch_size=config.source_loader.bs, 
                                         num_workers=config.source_loader.nw, sampler=source_train_sampler, 
                                         pin_memory=True)
    else:
        source_train_loader = DataLoader(dataset=source_train_set, batch_size=config.source_loader.bs, 
                                         num_workers=config.source_loader.nw, shuffle=True, 
                                         pin_memory=True)

    if config.target_dataset.dataset_type == "Cityscapes":
        if config.stage == 'self_train' or config.stage == "distill":
            target_train_set = Cityscapes(config, 'train', 'pseudo')
        else:
            target_train_set = Cityscapes(config, 'train', 'target')
        if multi_gpu:
            target_train_sampler = DistributedSampler(target_train_set, num_replicas=config.distribution.world_size, rank=rank)
            target_train_loader = DataLoader(dataset=target_train_set, batch_size=config.target_loader.bs,
                                             num_workers=config.target_loader.nw, sampler=target_train_sampler, 
                                             pin_memory=True)
        else:
            target_train_loader = DataLoader(dataset=target_train_set, batch_size=config.target_loader.bs,
                                             num_workers=config.target_loader.nw, shuffle=True, 
                                             pin_memory=True)
        target_val_set = Cityscapes(config, 'val', 'target')
        target_val_loader = DataLoader(dataset=target_val_set, batch_size=config.val_loader.bs, 
                                       num_workers=config.val_loader.nw, shuffle=False, 
                                       pin_memory=True)
    else:
        raise "Not implement target dataset"
    return source_train_loader, target_train_loader, target_val_loader