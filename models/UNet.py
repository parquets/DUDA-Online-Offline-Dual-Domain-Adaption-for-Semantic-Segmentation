import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones import UNetBackbone
from models.decoders import UNetDecoder
from models.predictors import UpsamplePredictor

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(UNet, self).__init__()
        self.backbone = UNetBackbone(in_channels=in_channels)
        self.decoder = UNetDecoder(out_channels=out_channels)
        self.predictor = nn.Upsample(scale_factor=2, mode='bilinear')
    
    def forward(self, data):
        backbone_out = self.backbone(data)
        decoder_out = self.decoder(backbone_out)
        predictor_out = self.predictor(decoder_out['deocoder_layer3'])
        output_dict = {
            'decoder_logits': predictor_out
        }
        return output_dict