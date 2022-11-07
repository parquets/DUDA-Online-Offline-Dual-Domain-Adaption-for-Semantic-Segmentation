import torch
import torch.nn as nn
import torch.nn.functional as F

class UpsamplePredictor(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(UpsamplePredictor, self).__init__()
        if isinstance(in_channels, list):
            in_channels = in_channels[-1]
        self.out_channels = [num_classes]
        self.num_classes = num_classes

    def forward(self, input, target) -> dict:
        assert isinstance(input, dict)
        for key in input:
            if 'logits' in key:
                input[key] = F.interpolate(input[key], size=(target.size()[2], target.size()[3]), mode='bilinear', align_corners=True)
        return input

