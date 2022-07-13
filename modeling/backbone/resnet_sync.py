# encoding: utf-8


import math

import torch
from torch import nn
# from spring.linklink.nn import SyncBatchNorm2d
# from spring.linklink.nn import syncbnVarMode_t

def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        self.bn1 = nn.SyncBatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.SyncBatchNorm(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.SyncBatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.SyncBatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.SyncBatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, last_stride=2, block=Bottleneck, layers=[3, 4, 6, 3]):
        self.inplanes = 64
        super().__init__()

        # global BN, bypass_bn_weight_list

        # def BNFunc(*args, **kwargs):
        #     return nn.SyncBatchNorm()
        #     # return SyncBatchNorm2d(*args, **kwargs,
        #     #                        group=bn_group,
        #     #                        sync_stats=bn_sync_stats,
        #     #                        var_mode=bn_var_mode)
        # if use_sync_bn:
        #     BN = BNFunc
        # else:
        #     BN = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.SyncBatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], dilation=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilation=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilation=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride, dilation=1)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.SyncBatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x, multi=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        if not multi:
            return x4
        else:
            return x4, [x3, x4]

    def load_param(self, model_path):
        param_dict = torch.load(model_path, 'cpu')
        if 'state_dict' in param_dict: param_dict = param_dict['state_dict']
        for i in param_dict:
            new_i = i
            if 'fc' in i or 'classifier' in new_i: continue
            if 'module.' == i[:7]: new_i = new_i[7:]
            if 'base.' == new_i[:5]: new_i = new_i[5:]
            if new_i not in self.state_dict().keys(): continue
            self.state_dict()[new_i].copy_(param_dict[i])

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, SyncBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()