import numpy as np
import torch
from torch import nn
from torch.nn import init
from math import floor

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class CBAMBlock(nn.Module):

    def __init__(self, channel=512, reduction=16, kernel_size=49):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual = x
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out + residual

# LOSSLESS POOLING ATTENTION
class LosslessPooling(torch.nn.Module):
    def __init__(self, kernel_size, dilation=1, padding=0,
                 stride=1, shuffle=True):
        super(LosslessPooling, self).__init__()
        self.kernel_size = kernel_size \
            if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.dilation = dilation
        self.padding = padding
        self.stride = stride
        self.shuffle = shuffle
        self.unfold = torch.nn.Unfold(kernel_size, dilation, padding, stride)

    def get_shape(self, h_w):
        h, w = list(map(lambda i: floor(((h_w[i] + (2*self.padding) - \
                                (self.dilation*(self.kernel_size[i]-1))-1)\
                                /self.stride) + 1),
                        range(2)))
        return h, w

    def forward(self, x):
        x_unf = self.unfold(x)
        x_out = x_unf.view(
            x.shape[0],
            x.shape[1] * self.kernel_size[0] * self.kernel_size[1],
            *self.get_shape(x.shape[2:])
        )
        if self.shuffle:
            return x_out[:, torch.randperm(x_out.shape[1])]
        return x_out

