import torch.nn as nn
from torch.utils.model_zoo import load_url as load_state_dict_from_url

from models.operators.IBN import IBN

affine_par = True

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, ibn=False):
        #  no ibn
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, affine=affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, ibn=False):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, stride=stride)
        self.bn1 = norm_layer(width, affine=affine_par)

        for i in self.bn1.parameters():
            i.requires_grad = False
            
        self.conv2 = conv3x3(width, width, 1, groups, dilation)
        self.bn2 = norm_layer(width, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.with_ibn = ibn
        if self.with_ibn:
            self.ibnc = IBN(width)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out) if not self.with_ibn else self.ibnc(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, strides=(1,2,2,2), dilations=(1,1,1,1),
                 norm_layer=None, with_ibn=False, freeze_bn=False, clr_bn=False):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64

        self.strides = strides
        self.dilations = dilations

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0],
                                       stride=strides[0], dilation=dilations[0],
                                       ibn=with_ibn)
        self.layer2 = self._make_layer(block, 128, layers[1],
                                       stride=strides[1], dilation=dilations[1])
        self.layer3 = self._make_layer(block, 256, layers[2],
                                       stride=strides[2], dilation=dilations[2])
        self.layer4 = self._make_layer(block, 512, layers[3],
                                       stride=strides[3], dilation=dilations[3])

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.out_channels = [64] + [x * block.expansion for x in [64, 128, 256, 512]]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                if not freeze_bn:
                    for i in m.parameters():
                        i.requires_grad = True

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
        
        self.clr_bn = clr_bn
        self.clr_bn_pretrain = None

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, ibn=False):
        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion, affine=affine_par),
            )
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False

        layers = []
        if ibn:
            ibn = False if planes == 512 else True
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, dilation, norm_layer, ibn=ibn))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=dilation,
                                norm_layer=norm_layer, ibn=ibn))

        return nn.Sequential(*layers)

    def forward(self, x) -> dict:
        output_dict = {}
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        output_dict['backbone_stem'] = x
        x = self.maxpool(x)

        x = self.layer1(x)
        output_dict['backbone_layer1'] = x


        x = self.layer2(x)
        output_dict['backbone_layer2'] = x


        x = self.layer3(x)
        output_dict['backbone_layer3'] = x


        x = self.layer4(x)
        if self.clr_bn_pretrain is not None:
            x = self.clr_bn_pretrain(x)
        output_dict['backbone_layer4'] = x

        return output_dict
    
    def update_clr_bn(self):
        self.clr_bn = True
        self.clr_bn_pretrain = nn.BatchNorm2d(2048, affine=affine_par)
