import torch
import torch.nn as nn
from torch.nn import modules
import torch.nn.functional as F


class UNetDecoder(nn.Module):
    def __init__(self, out_channels):
        super(UNetDecoder, self).__init__()

        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1, stride=1, bias=False),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
    def forward(self, data):
        assert isinstance(data, dict)
        backbone_layer0_out = data['backbone_layer0']        
        backbone_layer1_out = data['backbone_layer1']
        backbone_layer2_out = data['backbone_layer2']
        backbone_layer3_out = data['backbone_layer3']
        output_dict = {}
        # print(backbone_layer3_out.shape)
        x = self.layer0(backbone_layer3_out)
        output_dict['deocoder_layer0'] = x
        x = F.upsample(x, scale_factor=2, mode='bilinear') + backbone_layer2_out
        x = self.layer1(x)
        output_dict['deocoder_layer1'] = x
        x = F.upsample(x, scale_factor=2, mode='bilinear') + backbone_layer1_out
        x = self.layer2(x)
        output_dict['deocoder_layer2'] = x
        x = F.upsample(x, scale_factor=2, mode='bilinear') + backbone_layer0_out
        x = self.layer3(x)
        output_dict['deocoder_layer3'] = x
        return output_dict

if __name__ == '__main__':
    data = {
        'backbone_layer0': torch.randn((4,64,32,64)),
        'backbone_layer1': torch.randn((4,128,16,32)),
        'backbone_layer2': torch.randn((4,256,8,16)),
        'backbone_layer3': torch.randn((4,512,4,8)),
    }
    model = UNetDecoder(out_channels=1)
    output = model(data)
    for key in output:
        print(key)
        print(output[key].shape)
