import torch
import torch.nn as nn
from .aspp import ASPP_V2, ASPP_V2Plus
from .deeplabv2_decoder import DeepLabV2Decoder, DeepLabV2PlusDecoder, DeepLabV2PMultiDecoder
from .unet_decoder import UNetDecoder

def build_decoder(config, in_channels, out_channels) -> nn.Module:
    assert config.network.decoder_type is not None
    # print(config.network.decoder_type)
    decoder =  eval(config.network.decoder_type)(in_channels, out_channels, config.network.dropout_p)
    #print(decoder)
    assert isinstance(decoder, nn.Module)
    if config.network.decoder_fix_bn:
        for m in decoder.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
                for i in m.parameters():
                    i.requires_grad = False
    return decoder
    