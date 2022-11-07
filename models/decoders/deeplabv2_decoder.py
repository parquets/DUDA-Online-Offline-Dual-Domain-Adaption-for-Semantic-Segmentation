import torch
import torch.nn as nn
from .aspp import ASPP_V2, ASPP_V2Plus

class DeepLabV2Decoder(nn.Module):
    def __init__(self, input_channels, num_classes, drop_rate=0.0):
        super(DeepLabV2Decoder, self).__init__()
        self.dropout = nn.Dropout2d(drop_rate)
        self.aspp = ASPP_V2(input_channels[-1], [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.out_channels = [num_classes]

    def forward(self, features):
        assert isinstance(features, dict)
        output_dict = {}
        x = self.aspp(self.dropout(features['backbone_layer4']))
        output_dict['decoder_logits'] = x
        return output_dict

class DeepLabV2MultiDecoder(nn.Module):
    def __init__(self, input_channels, num_classes, drop_rate=0.5):
        super(DeepLabV2Decoder, self).__init__()
        self.primary_aspp = ASPP_V2(input_channels[-1], [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.auxiary_aspp = ASPP_V2(input_channels[-2], [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.out_channels = [num_classes]

    def forward(self, features):
        assert isinstance(features, dict)
        output_dict = {}
        auxiliary_out = self.auxiary_aspp(features['backbone_layer3'])
        primary_out = self.primary_aspp(features['backbone_layer4'])
        output_dict = {
            'auxiliary_decoder_logits': auxiliary_out,
            'primary_decoder_logits': primary_out
        }
        return output_dict

class DeepLabV2PlusDecoder(nn.Module):
    def __init__(self, input_channels, num_classes, drop_rate=0.5):
        super(DeepLabV2PlusDecoder, self).__init__()
        self.aspp = ASPP_V2Plus(input_channels[-1], [1, 6, 12, 18, 24], [1, 6, 12, 18, 24], 256)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=self.aspp.out_channels[-1], out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
        )
        self.head = nn.Sequential(
            nn.Dropout2d(p=0.1, inplace=True),
            nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1), bias=False),
        )
        self.out_channels = [num_classes]

        for m in self.bottleneck:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d) or isinstance(m, nn.GroupNorm) or isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in self.head:
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.001)

    def forward(self, features):
        assert isinstance(features, dict)
        output_dict = {}
        x = self.aspp(features['backbone_layer4'])
        x = self.bottleneck(x)
        output_dict['decoder_features'] = x
        x = self.head(x)
        output_dict['decoder_logits'] = x
        return output_dict

class DeepLabV2PMultiDecoder(nn.Module):
    def __init__(self, input_channels, num_classes, drop_rate=0.5):
        super(DeepLabV2PMultiDecoder, self).__init__()
        self.primary_aspp = ASPP_V2Plus(input_channels[-1], [1, 6, 12, 18, 24], [1, 6, 12, 18, 24], 256)
        self.primary_bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=self.primary_aspp.out_channels[-1], out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
        )
        self.primary_head = nn.Sequential(
            nn.Dropout2d(p=0.1, inplace=True),
            nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1), bias=False),
        )

        for m in self.primary_bottleneck:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d) or isinstance(m, nn.GroupNorm) or isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        for m in self.primary_head:
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.001)
        

        self.auxiliary_aspp = ASPP_V2Plus(input_channels[-2], [1, 6, 12, 18, 24], [1, 6, 12, 18, 24], 256)
        self.auxiliary_bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=self.auxiliary_aspp.out_channels[-1], out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
        )
        self.auxiliary_head = nn.Sequential(
            nn.Dropout2d(p=0.1, inplace=True),
            nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1), bias=False),
        )
        
        self.out_channels = [num_classes]

        for m in self.auxiliary_bottleneck:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d) or isinstance(m, nn.GroupNorm) or isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        for m in self.auxiliary_head:
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.001)
                
    def primary_forward(self, features):
        output_dict = {}
        x = self.primary_aspp(features)
        x = self.primary_bottleneck(x)
        output_dict['decoder_features'] = x
        x = self.primary_head(x)
        output_dict['decoder_logits'] = x
        return output_dict
    
    def auxiliary_forward(self, features):
        output_dict = {}
        x = self.auxiliary_aspp(features)

        x = self.auxiliary_bottleneck(x)
        output_dict['decoder_features'] = x
        x = self.auxiliary_head(x)
        output_dict['decoder_logits'] = x

        return output_dict

    def forward(self, features) -> dict:
        assert isinstance(features, dict)
        output_dict = {}
        primary_out_dict = self.primary_forward(features['backbone_layer4'])
        auxiliary_out_dict = self.auxiliary_forward(features['backbone_layer3'])
        output_dict = {
            'primary_decoder_features': primary_out_dict['decoder_features'],
            'primary_decoder_logits': primary_out_dict['decoder_logits'],
            'auxiliary_decoder_features': auxiliary_out_dict['decoder_features'],
            'auxiliary_decoder_logits': auxiliary_out_dict['decoder_logits'],
        }

        return output_dict

class DeepLabV2MultiTaskDecoder(nn.Module):
    def __init__(self, input_channels, num_classes, drop_rate=0.0):
        super(DeepLabV2MultiTaskDecoder, self).__init__()
        self.dropout = nn.Dropout2d(drop_rate)
        self.aspp = ASPP_V2(input_channels[-1], [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.out_channels = [num_classes]
        self.image_classfier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Linear(in_features=2048, out_features=num_classes)
        )

    def forward(self, features):
        assert isinstance(features, dict)
        output_dict = {}
        x = self.aspp(self.dropout(features['backbone_layer4']))
        output_dict['decoder_logits'] = x
        image_classfier_logits = self.image_classfier(features['backbone_layer4'])
        output_dict['image_classfier_logits'] = image_classfier_logits
        return output_dict