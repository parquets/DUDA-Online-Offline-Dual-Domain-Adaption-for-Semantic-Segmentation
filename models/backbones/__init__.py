import torch
import torch.nn as nn
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from .resnet import BasicBlock, Bottleneck
from .resnet import ResNet
from .unet import UNetBackbone


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}


def build_backbone(config) -> nn.Module:
    assert config.network.backbone_type is not None
    backbone_type = config.network.backbone_type
    # model = None
    if ("18" in backbone_type) or ("34" in backbone_type):
        model = ResNet(block=BasicBlock, layers=config.network.backbone_layers,
                       strides=config.network.backbone_strides, dilations=config.network.backbone_dilations,
                       freeze_bn=config.network.backbone_fix_bn, with_ibn=config.network.backbone_with_ibn)
    else:
        model = ResNet(block=Bottleneck, layers=config.network.backbone_layers,
                       strides=config.network.backbone_strides, dilations=config.network.backbone_dilations,
                       freeze_bn=config.network.backbone_fix_bn, with_ibn=config.network.backbone_with_ibn)
    
    state_dict = load_state_dict_from_url(model_urls[config.network.backbone_type],
                                              progress=True)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model