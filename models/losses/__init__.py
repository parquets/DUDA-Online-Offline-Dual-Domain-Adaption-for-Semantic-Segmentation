import imp
from .seg_loss import CrossEntropyLoss2d, SymmetricCrossEntropyLoss2d, FocalLoss2d
from .adv_loss import MSELoss
from .uda_loss import ImageWeightedMaxSquareloss, MaxSquareLoss