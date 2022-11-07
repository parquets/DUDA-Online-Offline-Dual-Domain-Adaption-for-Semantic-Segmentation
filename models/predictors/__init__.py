import torch
import torch.nn as nn
from .upsample_predictor import UpsamplePredictor

def build_predictor(config, in_channels, out_channels) -> nn.Module:
    assert config.network.predictor_type is not None
    return eval(config.network.predictor_type)(in_channels, out_channels)