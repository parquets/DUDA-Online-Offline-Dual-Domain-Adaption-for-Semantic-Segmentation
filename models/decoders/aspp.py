import torch
import torch.nn as nn


class ASPP_V2(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, outplanes):
        super(ASPP_V2, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out

class ASPP_V2Plus(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, outplanes):
        super(ASPP_V2Plus, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            conv_block = nn.Sequential(
                nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True),
                nn.BatchNorm2d(outplanes),
                nn.ReLU(inplace=True)
            )
            self.conv2d_list.append(conv_block)
        self.out_channels = [len(dilation_series)*outplanes]
        for m in self.conv2d_list:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out = torch.cat( (out, self.conv2d_list[i+1](x)), 1)
        return out

