from .GeneralSegmentor import GeneralSegmentor
from .discriminators import FCDiscriminator
from .UNet import UNet

from .losses import SymmetricCrossEntropyLoss2d, FocalLoss2d, CrossEntropyLoss2d


def build_model(config):
    pass

def build_loss(config):
    source_seg_loss = None
    target_seg_loss = None

    if config.train.source_loss == 'CrossEntropyLoss' or config.train.source_loss == 'CE':
        source_seg_loss = CrossEntropyLoss2d(ignore_index=255)
    elif config.train.source_loss == 'FocalLoss' or config.train.source_loss == 'FL':
        source_seg_loss = FocalLoss2d(ignore_index=255)
    elif config.train.source_loss == 'SymmetricCrossEntropyLoss' or config.train.source_loss == 'SCE':
        source_seg_loss = SymmetricCrossEntropyLoss2d(ignore_index=255)
    else:
        raise "Undefined source loss"
    
    if config.train.target_loss == 'CrossEntropyLoss' or config.train.target_loss == 'CE':
        target_seg_loss = CrossEntropyLoss2d(ignore_index=255)
    elif config.train.target_loss == 'FocalLoss' or config.train.target_loss == 'FL':
        target_seg_loss = FocalLoss2d(ignore_index=255)
    elif config.train.target_loss == 'SymmetricCrossEntropyLoss' or config.train.target_loss == 'SCE':
        target_seg_loss = SymmetricCrossEntropyLoss2d(ignore_index=255)
    else:
        raise "Undefined target loss"

    return source_seg_loss, target_seg_loss