# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Common modules
"""

import ast
import contextlib
import json
import math
import platform
import warnings
import zipfile
from collections import OrderedDict, namedtuple
from copy import copy
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp

from utils import TryExcept
from utils.dataloaders import exif_transpose, letterbox
from utils.general import (LOGGER, ROOT, Profile, check_requirements, check_suffix, check_version, colorstr,
                           increment_path, is_jupyter, make_divisible, non_max_suppression, scale_boxes, xywh2xyxy,
                           xyxy2xywh, yaml_load)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import copy_attr, smart_inference_mode
from mmcv.cnn import ConvModule
from mmengine.model import caffe2_xavier_init, constant_init

class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)
# 标准卷积：conv+BN+hardswish
class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()

        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))
# 这里的Conv1专为GSConv使用
class Conv1(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)

        self.act = SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class GSConv(nn.Module):
    # GSConv https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv1(c1, c_, k, s, None, g, act)
        self.cv2 = Conv1(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        # shuffle
        b, n, h, w = x2.data.size()
        b_n = b * n // 2
        y = x2.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)
        return torch.cat((y[0], y[1]), 1)


class GSBottleneck(nn.Module):
    # GS Bottleneck https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        c_ = int(c2 // 2)
        # for lighting
        self.conv_lighting = nn.Sequential(
            GSConv(c1, c_, 1, 1),
            GSConv(c_, c2, 1, 1, act=False))
        # for receptive field
        self.conv = nn.Sequential(
            GSConv(c1, c_, 3, 1),
            GSConv(c_, c2, 3, 1, act=False))
        self.shortcut = Conv(c1, c2, 3, 1, act=False)

    def forward(self, x):
        return self.conv_lighting(x) + self.shortcut(x)


class C2f_GS(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(GSBottleneck(self.c, self.c) for _ in range(n))
    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(self.cbam(torch.cat(y, 1)))



class ContextAggregation(nn.Module):
    """
    Context Aggregation Block.

    Args:
        in_channels (int): Number of input channels.
        reduction (int, optional): Channel reduction ratio. Default: 1.
        conv_cfg (dict or None, optional): Config dict for the convolution
            layer. Default: None.
    """

    def __init__(self, in_channels, reduction=1):
        super(ContextAggregation, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.inter_channels = max(in_channels // reduction, 1)

        conv_params = dict(kernel_size=1, act_cfg=None)

        self.a = ConvModule(in_channels, 1, **conv_params)
        self.k = ConvModule(in_channels, 1, **conv_params)
        self.v = ConvModule(in_channels, self.inter_channels, **conv_params)
        self.m = ConvModule(self.inter_channels, in_channels, **conv_params)

        self.init_weights()

    def init_weights(self):
        for m in (self.a, self.k, self.v):
            caffe2_xavier_init(m.conv)
        constant_init(self.m.conv, 0)

    def forward(self, x):
        #n, c = x.size(0)
        n = x.size(0)
        c = self.inter_channels
        #n, nH, nW, c = x.shape

        # a: [N, 1, H, W]
        a = self.a(x).sigmoid()

        # k: [N, 1, HW, 1]
        k = self.k(x).view(n, 1, -1, 1).softmax(2)

        # v: [N, 1, C, HW]
        v = self.v(x).view(n, 1, c, -1)

        # y: [N, C, 1, 1]
        y = torch.matmul(v, k).view(n, c, 1, 1)
        y = self.m(y) * a

        return x + y
class space_to_depth(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
         return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
#
# class GSConv(nn.Module):
#     # GSConv https://github.com/AlanLi1997/slim-neck-by-gsconv
#     # act参数在yolov7-tiny上记得修改为nn.LeakyReLU(0.1)
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
#         super().__init__()
#         c_ = c2 // 2
#         print("c_",c_,k,s,p,g,act)
#         self.cv1 = Conv(c1, c_, k, s, p, g, act)
#         self.cv2 = Conv(c_, c_, 5, 1, p, c_, act)
#
#     def forward(self, x):
#         x1 = self.cv1(x)
#         x2 = torch.cat((x1, self.cv2(x1)), 1)
#         # shuffle
#         # y = x2.reshape(x2.shape[0], 2, x2.shape[1] // 2, x2.shape[2], x2.shape[3])
#         # y = y.permute(0, 2, 1, 3, 4)
#         # return y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])
#
#         b, n, h, w = x2.size()
#         b_n = b * n // 2
#         y = x2.reshape(b_n, 2, h * w)
#         y = y.permute(1, 0, 2)
#         y = y.reshape(2, -1, n // 2, h, w)
#
#         return torch.cat((y[0], y[1]), 1)
#
# class GSBottleneck(nn.Module):
#     # GS Bottleneck https://github.com/AlanLi1997/slim-neck-by-gsconv
#     def __init__(self, c1, c2, k=3, s=1, e=0.5):
#         super().__init__()
#         c_ = int(c2*e)
#         # for lighting
#         self.conv_lighting = nn.Sequential(
#             GSConv(c1, c_, 1, 1),
#             GSConv(c_, c2, 3, 1, act=False))
#         self.shortcut = Conv(c1, c2, 1, 1, act=False)
#
#     def forward(self, x):
#         return self.conv_lighting(x) + self.shortcut(x)
#
# class GSBottleneckC(GSBottleneck):
#     # cheap GS Bottleneck https://github.com/AlanLi1997/slim-neck-by-gsconv
#     def __init__(self, c1, c2, k=3, s=1):
#         super().__init__(c1, c2, k, s)
#         self.shortcut = DWConv(c1, c2, k, s, act=False)
#
class VoVGSCSP(nn.Module):
    # VoVGSCSP module with GSBottleneck
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.gsb = nn.Sequential(*(GSBottleneck(c_, c_, e=1.0) for _ in range(n)))
        self.res = Conv(c_, c_, 3, 1, act=False)
        self.cv3 = Conv(2 * c_, c2, 1)  #


    def forward(self, x):
        x1 = self.gsb(self.cv1(x))
        y = self.cv2(x)
        return self.cv3(torch.cat((y, x1), dim=1))

class VoVGSCSPC(VoVGSCSP):
    # cheap VoVGSCSP module with GSBottleneck
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2)
        c_ = int(c2 * 0.5)  # hidden channels
        self.gsb = GSBottleneckC(c_, c_, 1, 1)

# 引入解耦头
class DecoupledHead(nn.Module):
    def __init__(self, ch=256, nc=80, anchors=()):
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.merge = Conv(ch, 256, 1, 1)
        self.cls_convs1 = Conv(256, 256, 3, 1, 1)
        self.cls_convs2 = Conv(256, 256, 3, 1, 1)
        self.reg_convs1 = Conv(256, 256, 3, 1, 1)
        self.reg_convs2 = Conv(256, 256, 3, 1, 1)
        self.cls_preds = nn.Conv2d(256, self.nc * self.na, 1)
        self.reg_preds = nn.Conv2d(256, 4 * self.na, 1)
        self.obj_preds = nn.Conv2d(256, 1 * self.na, 1)

    def forward(self, x):
        x = self.merge(x)
        x1 = self.cls_convs1(x)
        x1 = self.cls_convs2(x1)
        x1 = self.cls_preds(x1)
        x2 = self.reg_convs1(x)
        x2 = self.reg_convs2(x2)
        x21 = self.reg_preds(x2)
        x22 = self.obj_preds(x2)
        out = torch.cat([x21, x22, x1], 1)
        return out


# 为same卷积或same池化自动扩充
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p





# 深度可分离卷积
class DWConv(Conv):
    # Depth-wise convolution
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    # Depth-wise transpose convolution
    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)





class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class CrossConv(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class C3_cbam(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        self.cbam=CBAM(c2)
    def forward(self, x):
        return self.cv3(self.cbam(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1)))
class C3_se(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        self.se=SE(c2)
    def forward(self, x):
        return self.cv3(self.se(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1)))

class SimAM(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda


    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)
class C2f_SimAM(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(C2f_Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.sim=SimAM(1e-4)
    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(self.sim(torch.cat(y, 1)))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(self.sim(torch.cat(y, 1)))
class C2f_se(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(C2f_Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.se=SE((2 + n) * self.c)
    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(self.se(torch.cat(y, 1)))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(self.se(torch.cat(y, 1)))


class C2f_se2(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(C2f_Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.se=SE(c2)
    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.se(self.cv2(torch.cat(y, 1)))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.se(self.cv2(torch.cat(y, 1)))

class C2f_SimAM2(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(C2f_Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.sim=SimAM(1e-4)
    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.sim(self.cv2(torch.cat(y, 1)))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.sim(self.cv2(torch.cat(y, 1)))

class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        # print("x_h",x_h.shape)
        # print("x_w", x_w.shape)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

class C2f_cbam(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(C2f_Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.cbam=CBAM((2 + n) * self.c,7)
    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(self.cbam(torch.cat(y, 1)))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(self.cbam(torch.cat(y, 1)))

class C2f_ema(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(C2f_Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.ema = EMA((2 + n) * self.c)
    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(self.ema(torch.cat(y, 1)))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        # print("torch.cat(y, 1)",torch.cat(y, 1).shape)
        return self.cv2(torch.cat(y, 1))
class C2f_ema2(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(C2f_Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.ema = EMA(c2)
    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.ema(self.cv2(torch.cat(y, 1)))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        # print("torch.cat(y, 1)",torch.cat(y, 1).shape)
        return self.cv2(torch.cat(y, 1))

class CA_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA_Block, self).__init__()

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)

        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out

class C2f_ca(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(C2f_Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.ca = CA_Block((2 + n) * self.c)
    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(self.ca(torch.cat(y, 1)))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        # print("torch.cat(y, 1)",torch.cat(y, 1).shape)

        return self.cv2(torch.cat(y, 1))

class NAM(nn.Module):
    def __init__(self, channels, t=16):
        super(NAM, self).__init__()
        self.channels = channels
        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)

    def forward(self, x):
        residual = x
        x = self.bn2(x)
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = torch.sigmoid(x) * residual  #
        return x

class C2f_nam(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(C2f_Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.nam = NAM((2 + n) * self.c)
    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(self.nam(torch.cat(y, 1)))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
from torch.nn.parameter import Parameter
class sa_layer(nn.Module):
    """Constructs a Channel Spatial Group module.
    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, groups=64):
        super(sa_layer, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)
        return out
class C2f_sa(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(C2f_Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.sa = sa_layer((2 + n) * self.c)
    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(self.sa(torch.cat(y, 1)))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class ECA_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(ECA_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        out = x * y.expand_as(x)
        return out
class C2f_eca(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(C2f_Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.eca = ECA_block((2 + n) * self.c)
    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(self.eca(torch.cat(y, 1)))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
from efficientnet_pytorch.model import MemoryEfficientSwish

class AttnMap(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act_block = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0),
            MemoryEfficientSwish(),
            nn.Conv2d(dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.act_block(x)


class EfficientAttention(nn.Module):
    def __init__(self, dim, num_heads=8, group_split=[4, 4], kernel_sizes=[5], window_size=4,
                 attn_drop=0., proj_drop=0., qkv_bias=True):
        super().__init__()
        assert sum(group_split) == num_heads
        assert len(kernel_sizes) + 1 == len(group_split)
        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.scalor = self.dim_head ** -0.5
        self.kernel_sizes = kernel_sizes
        self.window_size = window_size
        self.group_split = group_split
        convs = []
        act_blocks = []
        qkvs = []
        # projs = []
        for i in range(len(kernel_sizes)):
            kernel_size = kernel_sizes[i]
            group_head = group_split[i]
            if group_head == 0:
                continue
            convs.append(nn.Conv2d(3 * self.dim_head * group_head, 3 * self.dim_head * group_head, kernel_size,
                                   1, kernel_size // 2, groups=3 * self.dim_head * group_head))
            act_blocks.append(AttnMap(self.dim_head * group_head))
            qkvs.append(nn.Conv2d(dim, 3 * group_head * self.dim_head, 1, 1, 0, bias=qkv_bias))
            # projs.append(nn.Linear(group_head*self.dim_head, group_head*self.dim_head, bias=qkv_bias))
        if group_split[-1] != 0:
            self.global_q = nn.Conv2d(dim, group_split[-1] * self.dim_head, 1, 1, 0, bias=qkv_bias)
            self.global_kv = nn.Conv2d(dim, group_split[-1] * self.dim_head * 2, 1, 1, 0, bias=qkv_bias)
            # self.global_proj = nn.Linear(group_split[-1]*self.dim_head, group_split[-1]*self.dim_head, bias=qkv_bias)
            self.avgpool = nn.AvgPool2d(window_size, window_size) if window_size != 1 else nn.Identity()

        self.convs = nn.ModuleList(convs)
        self.act_blocks = nn.ModuleList(act_blocks)
        self.qkvs = nn.ModuleList(qkvs)
        self.proj = nn.Conv2d(dim, dim, 1, 1, 0, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def high_fre_attntion(self, x: torch.Tensor, to_qkv: nn.Module, mixer: nn.Module, attn_block: nn.Module):
        '''
        x: (b c h w)
        '''
        b, c, h, w = x.size()
        qkv = to_qkv(x)  # (b (3 m d) h w)
        qkv = mixer(qkv).reshape(b, 3, -1, h, w).transpose(0, 1).contiguous()  # (3 b (m d) h w)
        q, k, v = qkv  # (b (m d) h w)
        attn = attn_block(q.mul(k)).mul(self.scalor)
        attn = self.attn_drop(torch.tanh(attn))
        res = attn.mul(v)  # (b (m d) h w)
        return res

    def low_fre_attention(self, x: torch.Tensor, to_q: nn.Module, to_kv: nn.Module, avgpool: nn.Module):
        '''
        x: (b c h w)
        '''
        b, c, h, w = x.size()

        q = to_q(x).reshape(b, -1, self.dim_head, h * w).transpose(-1, -2).contiguous()  # (b m (h w) d)
        kv = avgpool(x)  # (b c h w)
        kv = to_kv(kv).view(b, 2, -1, self.dim_head, (h * w) // (self.window_size ** 2)).permute(1, 0, 2, 4,
                                                                                                 3).contiguous()  # (2 b m (H W) d)
        k, v = kv  # (b m (H W) d)
        attn = self.scalor * q @ k.transpose(-1, -2)  # (b m (h w) (H W))
        attn = self.attn_drop(attn.softmax(dim=-1))
        res = attn @ v  # (b m (h w) d)
        res = res.transpose(2, 3).reshape(b, -1, h, w).contiguous()
        return res

    def forward(self, x: torch.Tensor):
        '''
        x: (b c h w)
        '''
        res = []
        for i in range(len(self.kernel_sizes)):
            if self.group_split[i] == 0:
                continue
            res.append(self.high_fre_attntion(x, self.qkvs[i], self.convs[i], self.act_blocks[i]))
        if self.group_split[-1] != 0:
            res.append(self.low_fre_attention(x, self.global_q, self.global_kv, self.avgpool))
        return self.proj_drop(self.proj(torch.cat(res, dim=1)))
class C2f_CloFormer(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(C2f_Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.glo = EfficientAttention((2 + n) * self.c)
    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(self.glo(torch.cat(y, 1)))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1 / 3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1 / 2 * (x_out11 + x_out21)
        return x_out
class C2f_Triplet(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(C2f_Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.tri = TripletAttention()
    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(self.tri(torch.cat(y, 1)))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class LSKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn
class C2f_LSK(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(C2f_Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.lsk = LSKblock((2 + n) * self.c)
    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(self.lsk(torch.cat(y, 1)))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))



# 模型轻量化
class DCNv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=1, dilation=1, groups=1, deformable_groups=1):
        super(DCNv2, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))

        out_channels_offset_mask = (self.deformable_groups * 3 *
                                    self.kernel_size[0] * self.kernel_size[1])
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            out_channels_offset_mask,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = Conv.default_act
        self.reset_parameters()

    def forward(self, x):
        offset_mask = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        x = torch.ops.torchvision.deform_conv2d(
            x,
            self.weight,
            offset,
            mask,
            self.bias,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1],
            self.groups,
            self.deformable_groups,
            True
        )
        x = self.bn(x)
        x = self.act(x)
        return x

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        std = 1. / math.sqrt(n)
        self.weight.data.uniform_(-std, std)
        self.bias.data.zero_()
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

class Bottleneck_DCN(nn.Module):
    # Standard bottleneck with DCN
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        if k[0] == 3:
            self.cv1 = DCNv2(c1, c_, k[0], 1)
        else:
            self.cv1 = Conv(c1, c_, k[0], 1)
        if k[1] == 3:
            self.cv2 = DCNv2(c_, c2, k[1], 1, groups=g)
        else:
            self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f_DCN(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_DCN(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
class GhostBottleneck1(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1,
                                                                            act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)
# 两个卷积conv或者两个卷积加上x
class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
class C2f_Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(C2f_Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        # print("y1",y.shape)
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
class C2f_Ghost(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(GhostBottleneck1(self.c, self.c) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C2f_RepGhost(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)

        self.m = nn.ModuleList(RepGhostBottleneck(self.c,self.c//2, self.c) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
class RepGhostBottleneck(nn.Module):
    """RepGhost bottleneck w/ optional SE"""

    def __init__(
        self,
        in_chs,
        mid_chs,
        out_chs,
        dw_kernel_size=3,
        stride=1,
        se_ratio=0.0,
        shortcut=True,
        reparam=True,
        reparam_bn=True,
        reparam_identity=False,
        deploy=False,
    ):
        super(RepGhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.0
        self.stride = stride
        self.enable_shortcut = shortcut
        self.in_chs = in_chs
        self.out_chs = out_chs

        # Point-wise expansion
        self.ghost1 = RepGhostModule(
            in_chs,
            mid_chs,
            relu=True,
            reparam_bn=reparam and reparam_bn,
            reparam_identity=reparam and reparam_identity,
            deploy=deploy,
        )
        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(
                mid_chs,
                mid_chs,
                dw_kernel_size,
                stride=stride,
                padding=(dw_kernel_size - 1) // 2,
                groups=mid_chs,
                bias=False,
            )
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = RepGhostModule(
            mid_chs,
            out_chs,
            relu=False,
            reparam_bn=reparam and reparam_bn,
            reparam_identity=reparam and reparam_identity,
            deploy=deploy,
        )
        # shortcut
        if in_chs == out_chs and self.stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_chs,
                    in_chs,
                    dw_kernel_size,
                    stride=stride,
                    padding=(dw_kernel_size - 1) // 2,
                    groups=in_chs,
                    bias=False,
                ),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(
                    in_chs, out_chs, 1, stride=1,
                    padding=0, bias=False,
                ),
                nn.BatchNorm2d(out_chs),
            )
    def forward(self, x):
        residual = x
       # 1st repghost bottleneck
        x1 = self.ghost1(x)
        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x1)
            x = self.bn_dw(x)
        else:
            x = x1
        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)
        # 2nd repghost bottleneck
        x = self.ghost2(x)
        if not self.enable_shortcut and self.in_chs == self.out_chs and self.stride == 1:
            return x
        return x + self.shortcut(residual)
class RepGhostModule(nn.Module):
    def __init__(
        self, inp, oup, kernel_size=1, dw_size=3, stride=1, relu=True, deploy=False, reparam_bn=True, reparam_identity=False
    ):
        super(RepGhostModule, self).__init__()
        init_channels = oup
        new_channels = oup
        self.deploy = deploy
        # print("参数",inp,init_channels,kernel_size,stride,kernel_size // 2)
        self.primary_conv = nn.Sequential(
            nn.Conv2d(
                inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False,
            ),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        fusion_conv = []
        fusion_bn = []
        if not deploy and reparam_bn:
            fusion_conv.append(nn.Identity())
            fusion_bn.append(nn.BatchNorm2d(init_channels))
        if not deploy and reparam_identity:
            fusion_conv.append(nn.Identity())
            fusion_bn.append(nn.Identity())

        self.fusion_conv = nn.Sequential(*fusion_conv)
        self.fusion_bn = nn.Sequential(*fusion_bn)

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(
                init_channels,
                new_channels,
                dw_size,
                1,
                dw_size // 2,
                groups=init_channels,
                bias=deploy,
            ),
            nn.BatchNorm2d(new_channels) if not deploy else nn.Sequential(),
            # nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        if deploy:
            self.cheap_operation = self.cheap_operation[0]
        if relu:
            self.relu = nn.ReLU(inplace=False)
        else:
            self.relu = nn.Sequential()

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        for conv, bn in zip(self.fusion_conv, self.fusion_bn):
            # print("x2",x2.shape)
            # print("bn(conv(x1))", bn(conv(x1)).shape)
            x2 = x2 + bn(conv(x1))
        return self.relu(x2)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.cheap_operation[0], self.cheap_operation[1])
        for conv, bn in zip(self.fusion_conv, self.fusion_bn):
            kernel, bias = self._fuse_bn_tensor(conv, bn, kernel3x3.shape[0], kernel3x3.device)
            kernel3x3 += self._pad_1x1_to_3x3_tensor(kernel)
            bias3x3 += bias
        return kernel3x3, bias3x3

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    @staticmethod
    def _fuse_bn_tensor(conv, bn, in_channels=None, device=None):
        in_channels = in_channels if in_channels else bn.running_mean.shape[0]
        device = device if device else bn.weight.device
        if isinstance(conv, nn.Conv2d):
            kernel = conv.weight
            assert conv.bias is None
        else:
            assert isinstance(conv, nn.Identity)
            kernel_value = np.zeros((in_channels, 1, 1, 1), dtype=np.float32)
            for i in range(in_channels):
                kernel_value[i, 0, 0, 0] = 1
            kernel = torch.from_numpy(kernel_value).to(device)

        if isinstance(bn, nn.BatchNorm2d):
            running_mean = bn.running_mean
            running_var = bn.running_var
            gamma = bn.weight
            beta = bn.bias
            eps = bn.eps
            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            return kernel * t, beta - running_mean * gamma / std
        assert isinstance(bn, nn.Identity)
        return kernel, torch.zeros(in_channels).to(kernel.device)

    def switch_to_deploy(self):
        if len(self.fusion_conv) == 0 and len(self.fusion_bn) == 0:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.cheap_operation = nn.Conv2d(in_channels=self.cheap_operation[0].in_channels,
                                         out_channels=self.cheap_operation[0].out_channels,
                                         kernel_size=self.cheap_operation[0].kernel_size,
                                         padding=self.cheap_operation[0].padding,
                                         dilation=self.cheap_operation[0].dilation,
                                         groups=self.cheap_operation[0].groups,
                                         bias=True)
        self.cheap_operation.weight.data = kernel
        self.cheap_operation.bias.data = bias
        self.__delattr__('fusion_conv')
        self.__delattr__('fusion_bn')
        self.fusion_conv = []
        self.fusion_bn = []
        self.deploy = True

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.0).clamp_(0.0, 6.0).div_(6.0)
    else:
        return F.relu6(x + 3.0) / 6.0
class SqueezeExcite(nn.Module):
    def __init__(
        self,
        in_chs,
        se_ratio=0.25,
        reduced_base_chs=None,
        act_layer=nn.ReLU,
        gate_fn=hard_sigmoid,
        divisor=4,
        **_,
    ):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible(
            (reduced_base_chs or in_chs) * se_ratio, divisor,
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x
class C3x(C3):
    # C3 module with cross-convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in -range(n)))


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3SPP(C3):
    # C3 module with SPP()
    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)


class C3Ghost(C3):
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1,
                                                                            act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(b, s, s, c // s ** 2, h, w)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(b, c // s ** 2, h * s, w * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)

# 三个分支concat操作
class BiFPN_Concat3(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN_Concat3, self).__init__()
        self.d = dimension
        # 设置可学习参数 nn.Parameter的作用是：将一个不可训练的类型Tensor转换成可以训练的类型parameter
        # 并且会向宿主模型注册该参数 成为其一部分 即model.parameters()会包含这个parameter
        # 从而在参数优化的时候可以自动一起优化
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        x = [weight[0] * x[0], weight[1] * x[1], weight[2] * x[2]]
        return torch.cat(x, self.d)

class Concat2(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


# 自适应融合模块
# class Concat3(nn.Module):
#     # Concatenate a list of tensors along dimension
#     def __init__(self, c1, c2, ratio=16, kernel_size=7, dimension=1):
#         super().__init__()
#         self.d = dimension  # 沿着哪个维度进行拼接
#         self.spatial_attention = SpatialAttention(7)
#         self.channel_attention = ChannelAttention(c1, ratio)
#
#     # 注x1为红外图像，x2为前景图像
#     def forward(self, x1, x2):
#         # 空间注意力生成权重
#         weight1 = self.spatial_attention(x1)
#         weight2 = self.spatial_attention(x2)
#         # print("weight1",weight1)
#         # print("weight2",weight2)
#         weight = (weight1 / weight2)
#         print("weights",weight)
#         # 为什么不能实现复杂场景下的自适应（部分自适应）
#         x1 = weight * x1
#         x2 = x2 * (2 - weight)
#         # 为什么是cat操作而不是add操作（为什么）
#         x = torch.cat((x1, x2), self.d)
#         # 这里进行修改，X应该是写错了
#         # X = self.channel_attention(x)
#         # print("输出的大小",x.shape)
#         # print("输出的通道注意力权重大小",X.shape)
#         X_w = self.channel_attention(x)
#         # print("X_W",X_w)
#         return x * X_w


# # 自适应融合模块
class Concat3(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, c1, c2, ratio=16, kernel_size=7, dimension=1):
        super().__init__()
        self.d = dimension  # 沿着哪个维度进行拼接
        self.spatial_attention = SpatialAttention(7)
        self.channel_attention = ChannelAttention(c1, ratio)

    # 注x1为红外图像，x2为前景图像
    def forward(self, x1, x2):
        # 空间注意力生成权重
        weight1 = self.spatial_attention(x1)
        weight2 = self.spatial_attention(x2)
        # print("weight1",weight1)
        # print("weight2",weight2)
        weight = (weight1 / weight2)
        # print("前景图像weights",weight)
        # 为什么不能实现复杂场景下的自适应（部分自适应）
        x2 = weight * x2
        # 红外图像权重为（）
        x1 = x1 * (2 - weight)
        print("红外图像权重weights", 2 - weight)
        # 为什么是cat操作而不是add操作（为什么）
        x = torch.cat((x1, x2), self.d)
        # 这里进行修改，X应该是写错了
        # X = self.channel_attention(x)
        # print("输出的大小",x.shape)
        # print("输出的通道注意力权重大小",X.shape)
        # X_w = self.channel_attention(x)
        # print("X_W",X_w)
        return x

class Concat4(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, c1, c2, ratio=16, kernel_size=7, dimension=1):
        super().__init__()
        self.d = dimension  # 沿着哪个维度进行拼接
        self.spatial_attention = SpatialAttention(7)
        self.channel_attention = ChannelAttention(c1, ratio)

    # 注x1为红外图像，x2为前景图像
    def forward(self, x1, x2):
        # 空间注意力生成权重
        weight1 = self.spatial_attention(x1)
        weight2 = self.spatial_attention(x2)
        # print("weight1",weight1)
        # print("weight2",weight2)
        weight = (weight1 / weight2)
        # print("红外weights",weight)
        # 为什么不能实现复杂场景下的自适应（部分自适应）
        # x2 = weight * x2
        # # 红外图像权重为（）
        # x1 = x1 * (2 - weight)
        x2 = x2 * (2 - weight)
        # 红外图像权重为（）
        x1 =weight* x1
        # print("前景图像weights", 2 - weight)
        # 为什么是cat操作而不是add操作（为什么）
        x = torch.cat((x1, x2), self.d)
        # 这里进行修改，X应该是写错了
        # X = self.channel_attention(x)
        # print("输出的大小",x.shape)
        # print("输出的通道注意力权重大小",X.shape)
        # X_w = self.channel_attention(x)
        # print("X_W",X_w)
        return x
class AC(nn.Module):
    def __init__(self, dim, reduction_ratio=32):
        super().__init__()
        self.ac = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction_ratio, 1),
            nn.Conv2d(dim // reduction_ratio, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return self.ac(x)


"""
dim 输入通道
reducition_ration变换压缩的通道数
"""


class ALS(nn.Module):
    def __init__(self, dim, reduction_ratio=32):
        super().__init__()
        self.als = nn.Sequential(
            nn.Conv2d(dim, dim // reduction_ratio, 1),
            nn.Conv2d(dim // reduction_ratio, dim // reduction_ratio, 3, 1, 1),
            nn.Conv2d(dim // reduction_ratio, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return self.als(x)


# class BA2M(nn.Module):
#     def __init__(self, dim, reduction_ratio=32, attn_drop=0., proj_drop=0.):
#         super().__init__()
#
#         self.ac = AC(dim, reduction_ratio)
#         self.als = ALS(dim, reduction_ratio)
#         # self.ags = AGS(dim, reduction_ratio, attn_drop, proj_drop)
#
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         # 一定要有dim=0
#         self.softmax = nn.Softmax(dim=0)
#         self.apply(self.init_weight)
#
#     #
#     def init_weight(self, m):
#         # print("m",m)
#         if isinstance(m, nn.BatchNorm2d):
#             nn.init.zeros_(m.bias)
#             nn.init.ones_(m.weight)
#         elif isinstance(m, (nn.Linear, nn.Conv2d)):
#             nn.init.xavier_normal_(m.weight.data)
#
#     def forward(self, x):
#
#         ac = self.ac(x)
#         als = self.avgpool(self.als(x))
#         # ags = self.avgpool(self.ags(x))
#
#         attn = torch.maximum(ac, als)
#         # attn = torch.maximum(attn, ags)
#         attn = torch.mean(attn, axis=1, keepdim=True)
#         attn = self.softmax(attn)
#         print(attn)
#         # x=[x1,x2],x1为动态图像，x2为红外图像
#         out = x * attn * 2
#         return out


class Dual_BA2M(nn.Module):
    def __init__(self, dim, reduction_ratio=32, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.ba2m = BA2M(dim=dim, reduction_ratio=reduction_ratio, attn_drop=attn_drop, proj_drop=proj_drop)
    # 注x1为动态图像，x2为红外图像
    def forward(self, x1, x2):
        for i in range(x1.shape[0]):
            output_x1 = torch.cat([x1[i:i + 1], x2[i:i + 1]], dim=0)
            output_x2 = self.ba2m(output_x1)
            output_x3 = torch.cat([output_x2[0:1], output_x2[1:2]], dim=1)
            if i == 0:
                output_x = output_x3
            else:
                output_x = torch.cat([output_x, output_x3], dim=0)
        return output_x
class AGS(nn.Module):
    def __init__(self, dim, reduction_ratio=32, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.reduction_ratio = reduction_ratio
        self.qkv = nn.Linear(dim // reduction_ratio, dim // reduction_ratio * 3)
        # 一定要有dim=,
        self.softmax = nn.Softmax(dim=0)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        # 获取B、C、H、W
        B, C, H, W = x.shape
        # print("x.shape", x.shape)
        # 展
        # x = torch.flatten(x, 2).transpose([0, 2, 1]).reshape((B, H * W, self.reduction_ratio, -1)) \
        #     .transpose([0, 2, 1, 3])  # B R H*W C' ,C' = C // R

        # 后宽高维度相乘
        x = torch.reshape(x, (B, C, H * W))
        # print("后宽高维度相乘",x.shape)
        # 后两维度交换
        x = x.permute(0, 2, 1)
        # print("后两维度交换", x.shape)
        # 重新定义大小
        x = x.reshape((B, H * W, self.reduction_ratio, -1))
        # print("重新定义大小", x.shape)
        # 最后中间两维交换位置
        x = x.permute(0, 2, 1, 3)
        # print("最后中间两维交换位置", x.shape)

        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, chunks=3, axis=-1)
        # print("q",q.shape)
        # print("k",k.shape)
        # print("v",v.shape)
        #
        # attn = q @ k.transpose([0, 1, 3, 2])
        attn = q @ k.permute(0, 1, 3, 2)
        # print("使用前",attn.shape)
        attn = self.softmax(attn)
        # print("使用后", attn.shape)
        attn = self.attn_drop(attn)
        #
        # out = (attn @ v).transpose([0, 2, 1, 3]).reshape((B, H * W, -1)).transpose([0, 1, 2]) \
        # .reshape((B, -1, H, W))
        out = (attn @ v).permute(0, 2, 1, 3).reshape((B, H * W, -1)).permute(0, 1, 2) \
            .reshape((B, -1, H, W))

        return out


class BA2M(nn.Module):
    def __init__(self, dim, reduction_ratio=32, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.ac = AC(dim, reduction_ratio)

        self.als = ALS(dim, reduction_ratio)
        self.ags = AGS(dim, reduction_ratio, attn_drop, proj_drop)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # 一定要有dim=0
        self.softmax = nn.Softmax(dim=0)

        self.apply(self.init_weight)

    def init_weight(self, m):
        # print("m",m)
        if isinstance(m, nn.BatchNorm2d):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        elif isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_normal_(m.weight.data)

    def forward(self, x):

        ac = self.ac(x)
        # print(ac)
        als = self.avgpool(self.als(x))
        # print(als)
        ags = self.avgpool(self.ags(x))

        attn = torch.maximum(ac, als)
        # print(attn)
        attn = torch.maximum(attn, ags)
        attn = torch.mean(attn, axis=1, keepdim=True)
        # print("attn",attn.shape)
        attn = self.softmax(attn)
        # print(attn)
        out = x * attn

        return out


class Multi_BA2M(nn.Module):
    def __init__(self, dim, reduction_ratio=32, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.ba2m = BA2M(dim=dim, reduction_ratio=reduction_ratio, attn_drop=attn_drop, proj_drop=proj_drop)

    # 注x1为动态图像，x2为红外图像
    def forward(self, x1, x2):
        # x = []
        # w = []
        for i in range(x1.shape[0]):
            output_x1 = torch.cat([x1[i:i + 1], x2[i:i + 1]], dim=0)
            # print("output1",output_x1.shape)
            output_x2 = self.ba2m(output_x1)
            # print("output2",output_x2.shape)
            output_x3 = torch.cat([output_x2[0:1], output_x2[1:2]], dim=1)
            # print("output3", output_x3.shape)
            if i == 0:
                output_x = output_x3
            else:
                output_x = torch.cat([output_x, output_x3], dim=0)
        return output_x



########################################################
###   1
class BA2M_c(nn.Module):
    def __init__(self, dim, reduction_ratio=32, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.ac = AC(dim, reduction_ratio)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # 一定要有dim=0
        self.softmax = nn.Softmax(dim=0)
        self.apply(self.init_weight)
    def init_weight(self, m):
        # print("m",m)
        if isinstance(m, nn.BatchNorm2d):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        elif isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_normal_(m.weight.data)
    def forward(self, x):
        # B*C*1*1
        ac = self.ac(x)
        attn = torch.mean(ac, axis=1, keepdim=True)
        # B*1*1*1
        attn = self.softmax(attn)
        # print("attn",attn)
        out = x * attn
        return out
class Dual_BA2M_c(nn.Module):
    def __init__(self, dim, reduction_ratio=32, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.ba2m = BA2M_c(dim=dim, reduction_ratio=reduction_ratio, attn_drop=attn_drop, proj_drop=proj_drop)
    # 注x1为动态图像，x2为红外图像
    def forward(self, x1, x2):
        # x = []
        # w = []
        for i in range(x1.shape[0]):
            output_x1 = torch.cat([x1[i:i + 1], x2[i:i + 1]], dim=0)
            # print("output1",output_x1.shape)
            output_x2 = self.ba2m(output_x1)
            # print("output2",output_x2.shape)
            output_x3 = torch.add(output_x2[0:1], output_x2[1:2])
            # print("output3", output_x3.shape)
            if i == 0:
                output_x = output_x3
            else:
                output_x = torch.cat([output_x, output_x3], dim=0)
        return output_x

class BA2M_s(nn.Module):
    def __init__(self, dim, reduction_ratio=32, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.als = ALS(dim, reduction_ratio)
        # self.ags = AGS(dim, reduction_ratio, attn_drop, proj_drop)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # 一定要有dim=0
        self.softmax = nn.Softmax(dim=0)
        self.apply(self.init_weight)
    def init_weight(self, m):
        # print("m",m)
        if isinstance(m, nn.BatchNorm2d):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        elif isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_normal_(m.weight.data)
    def forward(self, x):
        als = self.avgpool(self.als(x))
        attn = torch.mean(als, axis=1, keepdim=True)
        # B*1*1*1
        attn = self.softmax(attn)
        # print("attn",attn)
        out = x * attn
        return out
class Dual_BA2M_s(nn.Module):
    def __init__(self, dim, reduction_ratio=32, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.ba2m = BA2M_s(dim=dim, reduction_ratio=reduction_ratio, attn_drop=attn_drop, proj_drop=proj_drop)
    # 注x1为动态图像，x2为红外图像
    def forward(self, x1, x2):
        for i in range(x1.shape[0]):
            output_x1 = torch.cat([x1[i:i + 1], x2[i:i + 1]], dim=0)
            # print("output1",output_x1.shape)
            output_x2 = self.ba2m(output_x1)
            # print("output2",output_x2.shape)
            output_x3 = torch.add(output_x2[0:1], output_x2[1:2])
            # print("output3", output_x3.shape)
            if i == 0:
                output_x = output_x3
            else:
                output_x = torch.cat([output_x, output_x3], dim=0)
        return output_x

class BA2M_M(nn.Module):
    def __init__(self, dim, reduction_ratio=32, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.ac = AC(dim, reduction_ratio)
        self.als = ALS(dim, reduction_ratio)
        # self.ags = AGS(dim, reduction_ratio, attn_drop, proj_drop)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # 一定要有dim=0
        self.softmax = nn.Softmax(dim=0)
        self.apply(self.init_weight)
    def init_weight(self, m):
        # print("m",m)
        if isinstance(m, nn.BatchNorm2d):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        elif isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_normal_(m.weight.data)

    def forward(self, x):
        # B*C*1*1
        ac = self.ac(x)
        # B*C*1*1
        als = self.avgpool(self.als(x))
        # ags = self.avgpool(self.ags(x))
        # B*C*1*1
        attn = torch.maximum(ac, als)
        attn = torch.mean(attn, axis=1, keepdim=True)
        # B*1*1*1
        attn = self.softmax(attn)
        # print("attn",attn)
        out = x * attn
        return out


class Dual_BA2M_M(nn.Module):
    def __init__(self, dim, reduction_ratio=32, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.ba2m = BA2M_M(dim=dim, reduction_ratio=reduction_ratio, attn_drop=attn_drop, proj_drop=proj_drop)
    # 注x1为动态图像，x2为红外图像
    def forward(self, x1, x2):
        for i in range(x1.shape[0]):
            output_x1 = torch.cat([x1[i:i + 1], x2[i:i + 1]], dim=0)
            # print("output1",output_x1.shape)
            output_x2 = self.ba2m(output_x1)
            # print("output2",output_x2.shape)
            output_x3 = torch.add(output_x2[0:1], output_x2[1:2])
            # print("output3", output_x3.shape)
            if i == 0:
                output_x = output_x3
            else:
                output_x = torch.cat([output_x, output_x3], dim=0)
        return output_x
class Dual_BA2M_M_cat(nn.Module):
    def __init__(self, dim, reduction_ratio=32, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.ba2m = BA2M_M(dim=dim, reduction_ratio=reduction_ratio, attn_drop=attn_drop, proj_drop=proj_drop)
    # 注x1为动态图像，x2为红外图像
    def forward(self, x1, x2):
        for i in range(x1.shape[0]):
            output_x1 = torch.cat([x1[i:i + 1], x2[i:i + 1]], dim=0)
            # print("output1",output_x1.shape)
            output_x2 = self.ba2m(output_x1)
            # print("output2",output_x2.shape)
            output_x3 = torch.cat([output_x2[0:1], output_x2[1:2]], dim=1)
            # print("output3", output_x3.shape)
            if i == 0:
                output_x = output_x3
            else:
                output_x = torch.cat([output_x, output_x3], dim=0)
        return output_x

class SE(nn.Module):
    def __init__(self, dim, reduction_ratio=8):
        super().__init__()
        # 属性分配
        # 全局平均池化，输出的特征图的宽高=1
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        # 第一个全连接层将特征图的通道数下降4倍
        self.fc1 = nn.Linear(in_features=dim, out_features=dim // reduction_ratio, bias=False)
        # relu激活
        self.relu = nn.ReLU()
        # 第二个全连接层恢复通道数
        self.fc2 = nn.Linear(in_features=dim // reduction_ratio, out_features=dim, bias=False)
        # sigmoid激活函数，将权值归一化到0-1
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        # 获取输入特征图的shape
        b, c, h, w = inputs.shape
        # 全局平均池化 [b,c,h,w]==>[b,c,1,1]
        x = self.avg_pool(inputs)
        # 维度调整 [b,c,1,1]==>[b,c]
        x = x.view([b, c])

        # 第一个全连接下降通道 [b,c]==>[b,c//4]
        x = self.fc1(x)
        x = self.relu(x)
        # 第二个全连接上升通道 [b,c//4]==>[b,c]
        x = self.fc2(x)
        # 对通道权重归一化处理
        x = self.sigmoid(x)
        # print("权重输出x", x)
        # 调整维度 [b,c]==>[b,c,1,1]
        x = x.view([b, c, 1, 1])

        return x

class BA2M_SE(nn.Module):
    def __init__(self, dim, reduction_ratio=32, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.ac = SE(dim, reduction_ratio)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # 一定要有dim=0
        self.softmax = nn.Softmax(dim=0)
        self.apply(self.init_weight)


    def init_weight(self, m):
        # print("m",m)
        if isinstance(m, nn.BatchNorm2d):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        elif isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_normal_(m.weight.data)

    def forward(self, x):
        # B*C*1*1
        ac = self.ac(x)
        attn = torch.mean(ac, axis=1, keepdim=True)
        # B*1*1*1
        attn = self.softmax(attn)
        # print("attn",attn)
        out = x * attn
        return out
class Dual_BA2M_SE(nn.Module):
    def __init__(self, dim, reduction_ratio=32, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.ba2m = BA2M_SE(dim=dim, reduction_ratio=reduction_ratio, attn_drop=attn_drop, proj_drop=proj_drop)
    # 注x1为动态图像，x2为红外图像
    def forward(self, x1, x2):
        # x = []
        # w = []
        for i in range(x1.shape[0]):
            output_x1 = torch.cat([x1[i:i + 1], x2[i:i + 1]], dim=0)
            # print("output1",output_x1.shape)
            output_x2 = self.ba2m(output_x1)
            # print("output2",output_x2.shape)
            output_x3 = torch.add(output_x2[0:1], output_x2[1:2])
            # print("output3", output_x3.shape)
            if i == 0:
                output_x = output_x3
            else:
                output_x = torch.cat([output_x, output_x3], dim=0)

        return output_x


# 全局空间注意力R=2*1*H*W
class SpatialAttention_t(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention_t, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
# 近似通道注意力注意力模块(2*C*1*1)
class AC_AFF(nn.Module):
    def __init__(self, dim, reduction_ratio=8):
        super().__init__()
        inter_channels = int(dim // reduction_ratio)
        self.ac = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(dim),
        )
    def forward(self, x):
        return self.ac(x)


"""
dim 输入通道
reducition_ration变换压缩的通道数
"""
# 近似空间注意力模块(2*C*H*W)
class ALS_AFF(nn.Module):
    def __init__(self, dim, reduction_ratio=8):
        super().__init__()
        inter_channels = int(dim // reduction_ratio)
        self.als =  nn.Sequential(
            nn.Conv2d(dim, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(dim),
        )
    def forward(self, x):
        return self.als(x)

class BA2M_AFF(nn.Module):
    def __init__(self, dim, reduction_ratio=8, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.ac = AC_AFF(dim, reduction_ratio)
        self.als = ALS_AFF(dim, reduction_ratio)
        # self.ags = AGS(dim, reduction_ratio, attn_drop, proj_drop)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.ags = SpatialAttention_t(7)
        # 一定要有dim=0
        self.softmax = nn.Softmax(dim=0)
        self.apply(self.init_weight)

    def init_weight(self, m):
        # print("m",m)
        if isinstance(m, nn.BatchNorm2d):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        elif isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_normal_(m.weight.data)

    def forward(self, x):
        # B*C*1*1
        ac = self.ac(x)
        # B*C*1*1
        als = self.avgpool(self.als(x))
        # print(self.ags(x).shape)
        # ags0 = torch.mean(self.ags(x), dim=[2, 3], keepdim=True)
        # ags1,_ = torch.max(self.ags(x), 2, keepdim=True)
        # ags2,_ = torch.max(ags1 ,3, keepdim=True)
        # ags=ags2
        # # B*C*1*1
        attn = torch.maximum(ac, als)
        # # B*1*1*1
        attn = torch.mean(attn, axis=1, keepdim=True)
        # # # B*1*1*1
        attn = self.softmax(attn)
        # print("fusion26",attn)
        out = x *attn
        return out
# class BA2M_AFF(nn.Module):
#     def __init__(self, dim, reduction_ratio=8, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.ac = AC_AFF(dim, reduction_ratio)
#         self.als = ALS_AFF(dim, reduction_ratio)
#         # self.ags = AGS(dim, reduction_ratio, attn_drop, proj_drop)
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.ags = SpatialAttention_t(7)
#         # 一定要有dim=0
#         self.softmax = nn.Softmax(dim=0)
#         self.apply(self.init_weight)
#
#     def init_weight(self, m):
#         # print("m",m)
#         if isinstance(m, nn.BatchNorm2d):
#             nn.init.zeros_(m.bias)
#             nn.init.ones_(m.weight)
#         elif isinstance(m, (nn.Linear, nn.Conv2d)):
#             nn.init.xavier_normal_(m.weight.data)
#
#     def forward(self, x):
#         # # B*C*1*1
#         # ac = self.ac(x)
#         # # B*C*H*W
#         als=self.als(x)
#         # als1=self.avgpool(als)
#         ags2=torch.mean(als, dim=1, keepdim=True)
#
#
#         # als = self.avgpool(self.als(x))
#         # print(self.ags(x).shape)
#         ags=self.ags(x)
#         # ags0 = torch.mean(self.ags(x), dim=[2, 3], keepdim=True)
#         # ags1,_ = torch.max(self.ags(x), 2, keepdim=True)
#         # ags2,_ = torch.max(ags1 ,3, keepdim=True)
#         # ags=ags0+ags2
#         # # B*C*1*1
#         # attn1 = torch.maximum(ac, als1)
#         attn2 = torch.maximum(ags2,  ags)
#         # # # B*1*1*1
#         # attn1 = torch.mean(attn1, axis=1, keepdim=True)
#         attn2 = torch.mean(attn2, dim=[2, 3], keepdim=True)
#         attn=attn2
#         # # # B*1*1*1
#         attn = self.softmax(attn)
#         # print(attn)
#         out = x *attn
#         return out
#         # ac = self.ac(x)
# class BA2M_AFF(nn.Module):
#     def __init__(self, dim, reduction_ratio=8, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.ac = AC_AFF(dim, reduction_ratio)
#         self.als = ALS_AFF(dim, reduction_ratio)
#         # self.ags = AGS(dim, reduction_ratio, attn_drop, proj_drop)
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.ags = SpatialAttention_t(7)
#         # 一定要有dim=0
#         self.softmax = nn.Softmax(dim=0)
#         self.apply(self.init_weight)
#
#     def init_weight(self, m):
#         # print("m",m)
#         if isinstance(m, nn.BatchNorm2d):
#             nn.init.zeros_(m.bias)
#             nn.init.ones_(m.weight)
#         elif isinstance(m, (nn.Linear, nn.Conv2d)):
#             nn.init.xavier_normal_(m.weight.data)
#
#     def forward(self, x):
#         # # B*C*1*1
#         ac = self.ac(x)
#         # # B*C*H*W
#         # als=self.als(x)
#         # als1=self.avgpool(als)
#         # ags2=torch.mean(als, dim=1, keepdim=True)
#
#
#         als = self.avgpool(self.als(x))
#         # print(self.ags(x).shape)
#         # ags=self.ags(x)
#         ags0 = torch.mean(self.ags(x), dim=[2, 3], keepdim=True)
#         ags1,_ = torch.max(self.ags(x), 2, keepdim=True)
#         ags2,_ = torch.max(ags1 ,3, keepdim=True)
#         ags=ags0+ags2
#         # # B*C*1*1
#         attn1 = torch.maximum(ac, als)
#         # attn2 = torch.maximum(ags2,  ags)
#         # # # B*1*1*1
#         attn1 = torch.mean(attn1, axis=1, keepdim=True)
#         # attn2 = torch.mean(attn2, dim=[2, 3], keepdim=True)
#         # attn=attn1+ags
#         attn = torch.maximum(attn1, ags)
#         # # # B*1*1*1
#         attn = self.softmax(attn)
#         # print(attn)
#         out = x *attn
#         return out
#

# class BA2M_AFF_Concat3(nn.Module):
#     def __init__(self, dim, reduction_ratio=8, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.ac = AC_AFF(dim, reduction_ratio)
#         self.als = ALS_AFF(dim, reduction_ratio)
#         # self.ags = AGS(dim, reduction_ratio, attn_drop, proj_drop)
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.ags = SpatialAttention_t(7)
#         # 一定要有dim=0
#         self.softmax = nn.Softmax(dim=0)
#         self.apply(self.init_weight)
#
#     def init_weight(self, m):
#         # print("m",m)
#         if isinstance(m, nn.BatchNorm2d):
#             nn.init.zeros_(m.bias)
#             nn.init.ones_(m.weight)
#         elif isinstance(m, (nn.Linear, nn.Conv2d)):
#             nn.init.xavier_normal_(m.weight.data)
#
#     def forward(self, x):
#         # # B*C*1*1
#         # ac = self.ac(x)
#         # # # B*C*H*W
#         # als = self.avgpool(self.als(x))
#         # print(self.ags(x).shape)
#         # ags=self.ags(x)
#         ags0 = torch.mean(self.ags(x), dim=[2, 3], keepdim=True)
#         ags1,_ = torch.max(self.ags(x), 2, keepdim=True)
#         ags2,_ = torch.max(ags1 ,3, keepdim=True)
#         ags=ags0+ags2
#         # # B*C*1*1
#         # attn1 = torch.maximum(ac, als)
#         # attn2 = torch.maximum(ags2,  ags)
#         # # # B*1*1*1
#         # attn1 = torch.mean(attn1, axis=1, keepdim=True)
#         # attn2 = torch.mean(attn2, dim=[2, 3], keepdim=True)
#         # attn=attn1+ags
#         # attn = torch.maximum(attn1, ags)
#
#
#         # # # B*1*1*1
#         attn = self.softmax(ags)
#         # print(attn)
#         out = x *attn
#         return out

class BA2M_AFF_Concat3(nn.Module):
    def __init__(self, dim, reduction_ratio=8, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.ac = AC_AFF(dim, reduction_ratio)
        self.als = ALS_AFF(dim, reduction_ratio)
        # self.ags = AGS(dim, reduction_ratio, attn_drop, proj_drop)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.ags = SpatialAttention_t(7)
        # 一定要有dim=0
        self.softmax = nn.Softmax(dim=0)
        self.apply(self.init_weight)

    def init_weight(self, m):
        # print("m",m)
        if isinstance(m, nn.BatchNorm2d):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        elif isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_normal_(m.weight.data)

    def forward(self, x):
        # # B*C*1*1
        # ac = self.ac(x)
        # # # B*C*H*W
        # als = self.avgpool(self.als(x))
        # print(self.ags(x).shape)
        ags0=self.avgpool(self.ags(x))
        ags1=self.maxpool(self.ags(x))

        ags = ags0 + ags1
        # # B*C*1*1
        # # # B*1*1*1
        attn = self.softmax(ags)
        # print("concat3",attn)
        out = x * attn
        return out

class BA2M_AFF_Concat4(nn.Module):
    def __init__(self, dim, reduction_ratio=8, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.ac = AC_AFF(dim, reduction_ratio)
        self.als = ALS_AFF(dim, reduction_ratio)
        # self.ags = AGS(dim, reduction_ratio, attn_drop, proj_drop)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.ags = SpatialAttention_t(7)
        # 一定要有dim=0
        self.softmax = nn.Softmax(dim=0)
        self.apply(self.init_weight)

    def init_weight(self, m):
        # print("m",m)
        if isinstance(m, nn.BatchNorm2d):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        elif isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_normal_(m.weight.data)

    def forward(self, x):
        # # B*C*1*1
        # ac = self.ac(x)
        # # # B*C*H*W
        als = self.als(x)
        als1=torch.mean(als, axis=1, keepdim=True)

        # print(self.ags(x).shape)
        ags=self.ags(x)
        attn1 = torch.maximum(ags, als1)
        # attn2 = torch.maximum(ags, als1)
        # ags0=self.avgpool(self.ags(x))
        # ags1=self.maxpool(self.ags(x))
        attn=self.maxpool(attn1)
        # ags = ags0 + ags1
        # # B*C*1*1
        # # # B*1*1*1
        attn = self.softmax(attn)
        # print("concat4",attn)
        out = x * attn
        return out
class BA2M_AFF_Concat5(nn.Module):
    def __init__(self, dim, reduction_ratio=8, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.ac = AC_AFF(dim, reduction_ratio)
        self.als = ALS_AFF(dim, reduction_ratio)
        # self.ags = AGS(dim, reduction_ratio, attn_drop, proj_drop)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.ags = SpatialAttention_t(7)
        # 一定要有dim=0
        self.softmax = nn.Softmax(dim=0)
        self.apply(self.init_weight)

    def init_weight(self, m):
        # print("m",m)
        if isinstance(m, nn.BatchNorm2d):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        elif isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_normal_(m.weight.data)

    def forward(self, x):
        # # B*C*1*1
        ac = self.ac(x)
        # ac1,_ = torch.max(ac, axis=1, keepdim=True)
        ac1 = torch.mean(ac, axis=1, keepdim=True)
        # # # B*C*H*W
        als = self.als(x)
        als1=torch.mean(als, axis=1, keepdim=True)

        # print(self.ags(x).shape)
        ags=self.ags(x)
        attn1 = torch.maximum(ags, als1)

        # attn2 = torch.maximum(ags, als1)
        # ags0=self.avgpool(self.ags(x))
        # ags1=self.maxpool(self.ags(x))
        attn=self.maxpool(attn1)
        # attn = torch.maximum( attn, ac1)
        attn = ac1+attn
        # ags = ags0 + ags1
        # # B*C*1*1
        # # # B*1*1*1
        attn = self.softmax(attn)
        # print("concat5",attn)
        out = x * attn
        return out
# class BA2M_AFF_cancat3(nn.Module):
#     def __init__(self, dim, reduction_ratio=8, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.ac = AC_AFF(dim, reduction_ratio)
#         self.als = ALS_AFF(dim, reduction_ratio)
#         # self.ags = AGS(dim, reduction_ratio, attn_drop, proj_drop)
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         # 一定要有dim=0
#         self.softmax = nn.Softmax(dim=0)
#         self.apply(self.init_weight)
#
#     def init_weight(self, m):
#         # print("m",m)
#         if isinstance(m, nn.BatchNorm2d):
#             nn.init.zeros_(m.bias)
#             nn.init.ones_(m.weight)
#         elif isinstance(m, (nn.Linear, nn.Conv2d)):
#             nn.init.xavier_normal_(m.weight.data)
#
#     def forward(self, x):
#         ac = self.ac(x)
#         # B*C*1*1
#         als = self.avgpool(self.als(x))
#         # B*C*1*1
#         attn = ac+als
#         # B*1*1*1
#         attn = torch.mean(attn, axis=1, keepdim=True)
#         # B*1*1*1
#         attn = self.softmax(attn)
#         print(attn)
#         out = x * attn
#         return out



class Dual_BA2M_AFF(nn.Module):
    def __init__(self, dim, reduction_ratio=8, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.ba2m = BA2M_AFF(dim=dim, reduction_ratio=reduction_ratio, attn_drop=attn_drop, proj_drop=proj_drop)
    # 注x1为动态图像，x2为红外图像
    def forward(self, x1, x2):
        for i in range(x1.shape[0]):
            output_x1 = torch.cat([x1[i:i + 1], x2[i:i + 1]], dim=0)
            output_x2 = self.ba2m(output_x1)
            output_x3 = torch.add(output_x2[0:1], output_x2[1:2])
            if i == 0:
                output_x = output_x3
            else:
                output_x = torch.cat([output_x, output_x3], dim=0)
        return output_x


class Dual_BA2M_Concat3(nn.Module):
    def __init__(self, dim, reduction_ratio=8, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.ba2m = BA2M_AFF_Concat3(dim=dim, reduction_ratio=reduction_ratio, attn_drop=attn_drop, proj_drop=proj_drop)
    # 注x1为动态图像，x2为红外图像
    def forward(self, x1, x2):
        for i in range(x1.shape[0]):
            output_x1 = torch.cat([x1[i:i + 1], x2[i:i + 1]], dim=0)
            output_x2 = self.ba2m(output_x1)
            output_x3 = torch.add(output_x2[0:1], output_x2[1:2])
            if i == 0:
                output_x = output_x3
            else:
                output_x = torch.cat([output_x, output_x3], dim=0)
        return output_x
class Dual_BA2M_Concat4(nn.Module):
    def __init__(self, dim, reduction_ratio=8, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.ba2m = BA2M_AFF_Concat4(dim=dim, reduction_ratio=reduction_ratio, attn_drop=attn_drop, proj_drop=proj_drop)
    # 注x1为动态图像，x2为红外图像
    def forward(self, x1, x2):
        for i in range(x1.shape[0]):
            output_x1 = torch.cat([x1[i:i + 1], x2[i:i + 1]], dim=0)
            output_x2 = self.ba2m(output_x1)
            output_x3 = torch.add(output_x2[0:1], output_x2[1:2])
            if i == 0:
                output_x = output_x3
            else:
                output_x = torch.cat([output_x, output_x3], dim=0)
        return output_x
class Dual_BA2M_Concat5(nn.Module):
    def __init__(self, dim, reduction_ratio=8, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.ba2m = BA2M_AFF_Concat5(dim=dim, reduction_ratio=reduction_ratio, attn_drop=attn_drop, proj_drop=proj_drop)
    # 注x1为动态图像，x2为红外图像
    def forward(self, x1, x2):
        for i in range(x1.shape[0]):
            output_x1 = torch.cat([x1[i:i + 1], x2[i:i + 1]], dim=0)
            output_x2 = self.ba2m(output_x1)
            output_x3 = torch.add(output_x2[0:1], output_x2[1:2])
            if i == 0:
                output_x = output_x3
            else:
                output_x = torch.cat([output_x, output_x3], dim=0)
        return output_x
class Dual_BA2M_AFF_cat(nn.Module):
    def __init__(self, dim, reduction_ratio=8, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.ba2m = BA2M_AFF(dim=dim, reduction_ratio=reduction_ratio, attn_drop=attn_drop, proj_drop=proj_drop)
    # 注x1为动态图像，x2为红外图像
    def forward(self, x1, x2):
        # x = []
        # w = []
        for i in range(x1.shape[0]):
            output_x1 = torch.cat([x1[i:i + 1], x2[i:i + 1]], dim=0)
            # print("output1",output_x1.shape)
            output_x2 = self.ba2m(output_x1)
            # print("output2",output_x2.shape)
            output_x3 = torch.cat([output_x2[0:1], output_x2[1:2]], dim=1)
            if i == 0:
                output_x = output_x3
            else:
                output_x = torch.cat([output_x, output_x3], dim=0)

        return output_x


# 近似空间注意力模块(2*C*H*W)
class ALS_AFF_Spa(nn.Module):
    def __init__(self, dim, reduction_ratio=8):
        super().__init__()
        inter_channels = int(dim // reduction_ratio)
        self.als =  nn.Sequential(
            nn.Conv2d(dim, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(dim),
        )
    def forward(self, x):
        return self.als(x)
class BA2M_AFF_Spa(nn.Module):
    def __init__(self, dim, reduction_ratio=8, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.ac = AC_AFF(dim, reduction_ratio)
        self.als = ALS_AFF_Spa(dim, reduction_ratio)
        # self.ags = AGS(dim, reduction_ratio, attn_drop, proj_drop)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # 一定要有dim=0
        self.softmax = nn.Softmax(dim=0)
        self.apply(self.init_weight)

    def init_weight(self, m):
        # print("m",m)
        if isinstance(m, nn.BatchNorm2d):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        elif isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_normal_(m.weight.data)

    def forward(self, x):
        ac = self.ac(x)
        als = self.avgpool(self.als(x))
        # B*C*1*1
        attn = torch.maximum(ac, als)
        # B*1*1*1
        attn = torch.mean(attn, axis=1, keepdim=True)
        attn = self.softmax(attn)
        out = x * attn
        # print(attn)
        return out


class Dual_BA2M_AFF_Spa(nn.Module):
    def __init__(self, dim, reduction_ratio=8, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.ba2m = BA2M_AFF_Spa(dim=dim, reduction_ratio=reduction_ratio, attn_drop=attn_drop, proj_drop=proj_drop)
    # 注x1为动态图像，x2为红外图像
    def forward(self, x1, x2):
        # x = []
        # w = []
        for i in range(x1.shape[0]):
            output_x1 = torch.cat([x1[i:i + 1], x2[i:i + 1]], dim=0)
            output_x2 = self.ba2m(output_x1)
            output_x3 = torch.add(output_x2[0:1], output_x2[1:2])
            if i == 0:
                output_x = output_x3
            else:
                output_x = torch.cat([output_x, output_x3], dim=0)

        return output_x
#########################################################
class SKConv_dual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, M=2, r=16, L=32):
        '''
        :param in_channels:  输入通道维度
        :param out_channels: 输出通道维度   原论文中 输入输出通道维度相同
        :param stride:  步长，默认为1
        :param M:  分支数
        :param r: 特征Z的长度，计算其维度d 时所需的比率（论文中 特征S->Z 是降维，故需要规定 降维的下界）
        :param L:  论文中规定特征Z的下界，默认为32
        采用分组卷积： groups = 32,所以输入channel的数值必须是group的整数倍
        '''
        super(SKConv_dual, self).__init__()
        d = max(in_channels // r, L)  # 计算从向量C降维到 向量Z 的长度d
        self.M = M
        self.out_channels = out_channels
        self.conv = nn.ModuleList()  # 根据分支数量 添加 不同核的卷积操作
        for i in range(M):
            # 为提高效率，原论文中 扩张卷积5x5为 （3X3，dilation=2）来代替。 且论文中建议组卷积G=32
            self.conv.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride, padding=1 + i, dilation=1 + i, groups=32, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)))
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)  # 自适应pool到指定维度    这里指定为1，实现 GAP
        self.fc1 = nn.Sequential(nn.Conv2d(out_channels, d, 1, bias=False),
                                 nn.BatchNorm2d(d),
                                 nn.ReLU(inplace=True))  # 降维
        self.fc2 = nn.Conv2d(d, out_channels * M * 2, 1, 1, bias=False)  # 升维
        self.softmax = nn.Softmax(dim=1)  # 指定dim=1  使得两个全连接层对应位置进行softmax,保证 对应位置a+b+..=1

    def forward(self, input, input2):
        batch_size = input.size(0)
        # batch_size2 = input2.size(0)
        output = []
        # the part of split
        for i, conv in enumerate(self.conv):
            # print(i,conv(input).size())
            output.append(conv(input))  # [batch_size,out_channels,H,W]
            output.append(conv(input2))  # [batch_size,out_channels,H,W]
        # the part of fusion
        U = reduce(lambda x, y: x + y, output)  # 逐元素相加生成 混合特征U  [batch_size,channel,H,W]
        s = self.global_pool(U)  # [batch_size,channel,1,1]
        z = self.fc1(s)  # S->Z降维   # [batch_size,d,1,1]
        a_b = self.fc2(z)  # Z->a，b 升维  论文使用conv 1x1表示全连接。结果中前一半通道值为a,后一半为b   [batch_size,out_channels*M,1,1]
        a_b = a_b.reshape(batch_size, self.M * 2, self.out_channels,
                          -1)  # 调整形状，变为 两个全连接层的值[batch_size,M,out_channels,1]
        a_b = self.softmax(a_b)  # 使得两个全连接层对应位置进行softmax [batch_size,M,out_channels,1]
        # the part of selection
        a_b = list(a_b.chunk(self.M * 2,
                             dim=1))  # split to a and b   chunk为pytorch方法，将tensor按照指定维度切分成 几个tensor块 [[batch_size,1,out_channels,1],[batch_size,1,out_channels,1]
        a_b = list(map(lambda x: x.reshape(batch_size, self.out_channels, 1, 1),
                       a_b))  # 将所有分块  调整形状，即扩展两维  [[batch_size,out_channels,1,1],[batch_size,out_channels,1,1]
        # print("权重",a_b)
        V = list(map(lambda x, y: x * y, output,
                     a_b))  # 权重与对应  不同卷积核输出的U 逐元素相乘[batch_size,out_channels,H,W] * [batch_size,out_channels,1,1] = [batch_size,out_channels,H,W]
        V = reduce(lambda x, y: x + y,
                   V)  # 两个加权后的特征 逐元素相加  [batch_size,out_channels,H,W] + [batch_size,out_channels,H,W] = [batch_size,out_channels,H,W]
        return V  # [batch_size,out_channels,H,W]
# 只使用
# 近似通道注意力注意力模块(2*C*1*1)
class AC_H(nn.Module):
    def __init__(self, dim, reduction_ratio=8):
        super().__init__()
        inter_channels = int(dim // reduction_ratio)
        self.ac = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            nn.Conv2d(dim, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(dim),
        )
    def forward(self, x):
        return self.ac(x)


"""
dim 输入通道
reducition_ration变换压缩的通道数
"""
# 近似空间注意力模块(2*C*H*W)
class ALS_W(nn.Module):
    def __init__(self, dim, reduction_ratio=8):
        super().__init__()
        inter_channels = int(dim // reduction_ratio)
        self.als =  nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)),
            nn.Conv2d(dim, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(dim),
        )
    def forward(self, x):
        return self.als(x)

class BA2M_HW(nn.Module):
    def __init__(self, dim, reduction_ratio=8, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.ac = AC_H(dim, reduction_ratio)
        self.als = ALS_W(dim, reduction_ratio)
        # self.ags = AGS(dim, reduction_ratio, attn_drop, proj_drop)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # 一定要有dim=0
        self.softmax = nn.Softmax(dim=0)
        self.apply(self.init_weight)

    def init_weight(self, m):
        # print("m",m)
        if isinstance(m, nn.BatchNorm2d):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        elif isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_normal_(m.weight.data)

    def forward(self, x):
        # print("x",x.shape)
        ac = self.ac(x)
        # ac=self.avgpool(self.ac(x))
        # B*C*1*1
        # als = self.avgpool(self.als(x))
        als=self.als(x)
        # B*C*1*1
        attn = torch.maximum(ac, als)
        # B*1*1*1
        attn = torch.mean(attn, axis=1, keepdim=True)
        attn=self.avgpool(attn)
        # B*1*1*1
        attn = self.softmax(attn)
        # print("attn",attn.shape)
        # print(attn)
        out = x * attn
        return out


class Dual_BA2M_HW(nn.Module):
    def __init__(self, dim, reduction_ratio=8, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.ba2m = BA2M_HW(dim=dim, reduction_ratio=reduction_ratio, attn_drop=attn_drop, proj_drop=proj_drop)
    # 注x1为动态图像，x2为红外图像
    def forward(self, x1, x2):
        # x = []
        # w = []
        for i in range(x1.shape[0]):
            output_x1 = torch.cat([x1[i:i + 1], x2[i:i + 1]], dim=0)
            # print("output1",output_x1.shape)
            output_x2 = self.ba2m(output_x1)
            # print("output2",output_x2.shape)
            output_x3 = torch.add(output_x2[0:1], output_x2[1:2])
            # print("output3", output_x3.shape)
            if i == 0:
                output_x = output_x3
            else:
                output_x = torch.cat([output_x, output_x3], dim=0)

        return output_x

################## SKConv
from functools import reduce


class SKConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, M=2, r=16, L=32):
        '''
        :param in_channels:  输入通道维度
        :param out_channels: 输出通道维度   原论文中 输入输出通道维度相同
        :param stride:  步长，默认为1
        :param M:  分支数
        :param r: 特征Z的长度，计算其维度d 时所需的比率（论文中 特征S->Z 是降维，故需要规定 降维的下界）
        :param L:  论文中规定特征Z的下界，默认为32
        采用分组卷积： groups = 32,所以输入channel的数值必须是group的整数倍
        '''
        super(SKConv, self).__init__()
        d = max(in_channels // r, L)  # 计算从向量C降维到 向量Z 的长度d
        self.M = M
        self.out_channels = out_channels
        self.conv = nn.ModuleList()  # 根据分支数量 添加 不同核的卷积操作
        for i in range(M):
            # 为提高效率，原论文中 扩张卷积5x5为 （3X3，dilation=2）来代替。 且论文中建议组卷积G=32
            self.conv.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride, padding=1 + i, dilation=1 + i, groups=32, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)))
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)  # 自适应pool到指定维度    这里指定为1，实现 GAP
        self.fc1 = nn.Sequential(nn.Conv2d(out_channels, d, 1, bias=False),
                                 nn.BatchNorm2d(d),
                                 nn.ReLU(inplace=True))  # 降维
        self.fc2 = nn.Conv2d(d, out_channels * M, 1, 1, bias=False)  # 升维
        self.softmax = nn.Softmax(dim=1)  # 指定dim=1  使得两个全连接层对应位置进行softmax,保证 对应位置a+b+..=1

    def forward(self, input):
        batch_size = input.size(0)
        output = []
        # the part of split
        for i, conv in enumerate(self.conv):
            # print(i,conv(input).size())
            output.append(conv(input))  # [batch_size,out_channels,H,W]

        # the part of fusion
        U = reduce(lambda x, y: x + y, output)  # 逐元素相加生成 混合特征U  [batch_size,channel,H,W]
        # print(U.size())
        s = self.global_pool(U)  # [batch_size,channel,1,1]
        # print(s.size())
        z = self.fc1(s)  # S->Z降维   # [batch_size,d,1,1]
        # print(z.size())
        a_b = self.fc2(z)  # Z->a，b 升维  论文使用conv 1x1表示全连接。结果中前一半通道值为a,后一半为b   [batch_size,out_channels*M,1,1]
        # print(a_b.size())
        a_b = a_b.reshape(batch_size, self.M, self.out_channels, -1)  # 调整形状，变为 两个全连接层的值[batch_size,M,out_channels,1]
        # print(a_b.size())
        a_b = self.softmax(a_b)  # 使得两个全连接层对应位置进行softmax [batch_size,M,out_channels,1]

        # the part of selection
        a_b = list(a_b.chunk(self.M,
                             dim=1))  # split to a and b   chunk为pytorch方法，将tensor按照指定维度切分成 几个tensor块 [[batch_size,1,out_channels,1],[batch_size,1,out_channels,1]
        # print(a_b[0].size())
        # print(a_b[1].size())
        a_b = list(map(lambda x: x.reshape(batch_size, self.out_channels, 1, 1),
                       a_b))  # 将所有分块  调整形状，即扩展两维  [[batch_size,out_channels,1,1],[batch_size,out_channels,1,1]
        print("a_b", a_b)
        V = list(map(lambda x, y: x * y, output,
                     a_b))  # 权重与对应  不同卷积核输出的U 逐元素相乘[batch_size,out_channels,H,W] * [batch_size,out_channels,1,1] = [batch_size,out_channels,H,W]
        V = reduce(lambda x, y: x + y,
                   V)  # 两个加权后的特征 逐元素相加  [batch_size,out_channels,H,W] + [batch_size,out_channels,H,W] = [batch_size,out_channels,H,W]
        return V  # [batch_size,out_channels,H,W]

class SKConv2(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,M=2,r=16,L=32):
        '''
        :param in_channels:  输入通道维度
        :param out_channels: 输出通道维度   原论文中 输入输出通道维度相同
        :param stride:  步长，默认为1
        :param M:  分支数
        :param r: 特征Z的长度，计算其维度d 时所需的比率（论文中 特征S->Z 是降维，故需要规定 降维的下界）
        :param L:  论文中规定特征Z的下界，默认为32
        采用分组卷积： groups = 32,所以输入channel的数值必须是group的整数倍
        '''
        super(SKConv2,self).__init__()
        d=max(in_channels//r,L)   # 计算从向量C降维到 向量Z 的长度d
        self.M=M
        self.out_channels=out_channels
        self.conv=nn.ModuleList()  # 根据分支数量 添加 不同核的卷积操作
        for i in range(M):
            # 为提高效率，原论文中 扩张卷积5x5为 （3X3，dilation=2）来代替。 且论文中建议组卷积G=32
            self.conv.append(nn.Sequential(nn.Conv2d(in_channels,out_channels,3,stride,padding=1+i,dilation=1+i,groups=32,bias=False),
                                           nn.BatchNorm2d(out_channels),
                                           nn.ReLU(inplace=True)))
        self.global_pool=nn.AdaptiveAvgPool2d(output_size = 1) # 自适应pool到指定维度    这里指定为1，实现 GAP
        self.fc1=nn.Sequential(nn.Conv2d(out_channels,d,1,bias=False),
                               nn.BatchNorm2d(d),
                               nn.ReLU(inplace=True))   # 降维
        self.fc2=nn.Conv2d(d,out_channels*M,1,1,bias=False)  # 升维
        self.softmax=nn.Softmax(dim=1) # 指定dim=1  使得两个全连接层对应位置进行softmax,保证 对应位置a+b+..=1
    def forward(self, input1,input2):
        # print("input:", input.shape)
        batch_size=input1.size(0)
        # output=[]
        # print("x:",output[0].shape)
        # print("y:", output[1].shape)
        #the part of fusion
        output = [input1,input2]
        U=reduce(lambda x,y:x+y,output) # 逐元素相加生成 混合特征U  [batch_size,channel,H,W]
        # print(U.size())
        s=self.global_pool(U)     # [batch_size,channel,1,1]
        # print(s.size())
        z=self.fc1(s)  # S->Z降维   # [batch_size,d,1,1]
        # print(z.size())
        a_b=self.fc2(z) # Z->a，b 升维  论文使用conv 1x1表示全连接。结果中前一半通道值为a,后一半为b   [batch_size,out_channels*M,1,1]
        # print(a_b.size())
        a_b=a_b.reshape(batch_size,self.M,self.out_channels,-1) #调整形状，变为 两个全连接层的值[batch_size,M,out_channels,1]
        # print(a_b.size())
        a_b=self.softmax(a_b) # 使得两个全连接层对应位置进行softmax [batch_size,M,out_channels,1]
        #the part of selection
        a_b=list(a_b.chunk(self.M,dim=1))#split to a and b   chunk为pytorch方法，将tensor按照指定维度切分成 几个tensor块 [[batch_size,1,out_channels,1],[batch_size,1,out_channels,1]
        a_b=list(map(lambda x:x.reshape(batch_size,self.out_channels,1,1),a_b)) # 将所有分块  调整形状，即扩展两维  [[batch_size,out_channels,1,1],[batch_size,out_channels,1,1]
        V=list(map(lambda x,y:x*y,output,a_b)) # 权重与对应  不同卷积核输出的U 逐元素相乘[batch_size,out_channels,H,W] * [batch_size,out_channels,1,1] = [batch_size,out_channels,H,W]
        V=reduce(lambda x,y:x+y,V) # 两个加权后的特征 逐元素相加  [batch_size,out_channels,H,W] + [batch_size,out_channels,H,W] = [batch_size,out_channels,H,W]
        return V    # [batch_size,out_channels,H,W]
# （1）通道注意力机制

class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print("self.pool(x)",self.fc(self.pool(x)).shape)
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):  # ch_in, kernels
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))


# 残差+SE结果
class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=4):  # inplanes, planes输入输出通道数
        super(SEBottleneck, self).__init__()

        self.firconv = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
        )
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 1, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 1)
        self.relu = nn.ReLU(inplace=True)
        self.se = se_block(planes * 1, reduction)
        # self.se = cbam(planes * 1, ratio=4, kernel_size=7)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # 先经过一层1*1卷积
        out_x = self.firconv(x)
        residual = out_x

        # print(x.shape)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        #     print("下采样图像大小", residual)
        # print("通过注意力机制大小", out.shape)
        out += residual
        out = self.relu(out)
        return out


# 残差+SE结果
class CABMBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=4):  # inplanes, planes输入输出通道数
        super(CABMBottleneck, self).__init__()

        self.firconv = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
        )
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 1, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 1)
        self.relu = nn.ReLU(inplace=True)
        # self.se = se_block(planes * 1, reduction)
        self.cbam = CBAM(planes * 1, ratio=4, kernel_size=7)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # 先经过一层1*1卷积
        out_x = self.firconv(x)
        residual = out_x

        # print(x.shape)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        # print("se前", out.shape)
        out = self.cbam(out)
        # print("se后", out.shape)
        if self.downsample is not None:
            residual = self.downsample(x)
        #     print("下采样图像大小", residual)
        # print("通过注意力机制大小", out.shape)
        out += residual
        out = self.relu(out)
        return out


# 残差+SE结果
class EIR(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, middle_channel=16, stride=1, downsample=None,
                 reduction=16):  # inplanes, planes输入输出通道数
        super(EIR, self).__init__()

        self.firconv = nn.Sequential(
            nn.Conv2d(inplanes, middle_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(middle_channel),
        )
        self.SEblock = SEBottleneck(inplanes=middle_channel, planes=planes, stride=stride, downsample=downsample,
                                    reduction=reduction)

    def forward(self, x):
        # 先经过一层1*1卷积
        out_x = self.firconv(x)
        # print(out_x.shape)
        output = self.SEblock(out_x)
        return output


# 残差+SE结果
class EIR_cabm(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, middle_channel=16, stride=1, downsample=None,
                 reduction=16):  # inplanes, planes输入输出通道数
        super(EIR_cabm, self).__init__()

        self.firconv = nn.Sequential(
            nn.Conv2d(inplanes, middle_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(middle_channel),
        )
        self.CABMblock = CABMBottleneck(inplanes=middle_channel, planes=planes, stride=stride, downsample=downsample,
                                        reduction=reduction)

    def forward(self, x):
        # 先经过一层1*1卷积
        out_x = self.firconv(x)
        # print(out_x.shape)
        output = self.CABMblock(out_x)
        return output


# 结合BiFPN 设置可学习参数 学习不同分支的权重
# 两个分支add操作
class BiFPN_Add2(nn.Module):
    def __init__(self, c1, c2):
        super(BiFPN_Add2, self).__init__()
        # 设置可学习参数 nn.Parameter的作用是：将一个不可训练的类型Tensor转换成可以训练的类型parameter
        # 并且会向宿主模型注册该参数 成为其一部分 即model.parameters()会包含这个parameter
        # 从而在参数优化的时候可以自动一起优化
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.conv = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0)
        self.silu = nn.SiLU()

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        return self.conv(self.silu(weight[0] * x[0] + weight[1] * x[1]))


# 三个分支add操作
class BiFPN_Add3(nn.Module):
    def __init__(self, c1, c2):
        super(BiFPN_Add3, self).__init__()
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.conv = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0)
        self.silu = nn.SiLU()

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        return self.conv(self.silu(weight[0] * x[0] + weight[1] * x[1] + weight[2] * x[2]))


# 结合BiFPN 设置可学习参数 学习不同分支的权重
# 两个分支concat操作
class BiFPN_Concat2(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN_Concat2, self).__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        # print("BiFPN_Concat2",x.shape)
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        x = [weight[0] * x[0], weight[1] * x[1]]
        return torch.cat(x, self.d)


# 三个分支concat操作
class BiFPN_Concat3(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN_Concat3, self).__init__()
        self.d = dimension
        # 设置可学习参数 nn.Parameter的作用是：将一个不可训练的类型Tensor转换成可以训练的类型parameter
        # 并且会向宿主模型注册该参数 成为其一部分 即model.parameters()会包含这个parameter
        # 从而在参数优化的时候可以自动一起优化
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        x = [weight[0] * x[0], weight[1] * x[1], weight[2] * x[2]]
        return torch.cat(x, self.d)


###############  AFF
class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels=64, r=4):
        super().__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.conv1 = nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0)
        # self.bn1 = nn.BatchNorm2d(inter_channels)
        # self.relu = nn.ReLU(inplace=True)
        # self.conv2 = nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0)
        # self.bn2 = nn.BatchNorm2d(channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        # print("xl大小", xl.shape)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo
class AFF_HW(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels=64, r=4):
        super(AFF_HW, self).__init__()
        inter_channels = int(channels // r)

        # 局部特征模块B*C*1*1
        self.local_att = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # 全局特征模块B*C*H*W
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        # print("x的大小", x.shape)
        # print("residual大小", residual.shape)
        xa = x + residual
        # print("xa大小", xa.shape)
        xl = self.local_att(xa)
        # print("xl大小", xl.shape)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        # print("权重wei", wei.shape)
        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo
class AFF_CHW(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels=64, r=4):
        super(AFF_CHW, self).__init__()
        inter_channels = int(channels // r)

        # 局部特征模块B*C*1*1
        self.local_att = nn.Sequential(

            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # 全局特征模块B*C*H*W
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # 局部特征模块B*C*H*1
        self.local_H = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # 全局特征模块B*C*H*W
        self.local_W = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        # print("x的大小", x.shape)
        # print("residual大小", residual.shape)
        xa = x + residual
        # print("xa大小", xa.shape)
        xl = self.local_att(xa)
        # print("xl大小", xl.shape)
        xg = self.global_att(xa)

        # print("xg大小",xg.shape)
        xh = self.local_H(xa)
        # print("xh大小", xh.shape)
        xw = self.local_W(xa)
        # print("xw大小", xw.shape)

        xlg = xl + xg+xh+xw
        wei = self.sigmoid(xlg)
        # print("权重wei", wei.shape)
        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo

class AFF_CHW2(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels=64, r=4):
        super(AFF_CHW2, self).__init__()
        inter_channels = int(channels // r)

        # # 局部特征模块B*C*1*1
        # self.local_att = nn.Sequential(
        #
        #     nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(inter_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(channels),
        # )
        # 全局特征模块B*C*H*W
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # 局部特征模块B*C*H*1
        self.local_H = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # 全局特征模块B*C*H*W
        self.local_W = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        # print("x的大小", x.shape)
        # print("residual大小", residual.shape)
        xa = x + residual
        # print("xa大小", xa.shape)
        # xl = self.local_att(xa)
        # print("xl大小", xl.shape)
        xg = self.global_att(xa)

        # print("xg大小",xg.shape)
        xh = self.local_H(xa)
        # print("xh大小", xh.shape)
        xw = self.local_W(xa)
        # print("xw大小", xw.shape)
        xlg =  xg+xh+xw
        wei = self.sigmoid(xlg)
        # print("权重wei", wei.shape)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo
class AFF_CHW3(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels=64, r=4):
        super(AFF_CHW3, self).__init__()
        inter_channels = int(channels // r)

        # 局部特征模块B*C*1*1
        self.local_att = nn.Sequential(

            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # 局部特征模块B*C*H*1
        self.local_H = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # 全局特征模块B*C*H*W
        self.local_W = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        # print("x的大小", x.shape)
        # print("residual大小", residual.shape)
        xa = x + residual
        # print("xa大小", xa.shape)
        xl = self.local_att(xa)
        # print("xl大小", xl.shape)
        # xg = self.global_att(xa)

        # print("xg大小",xg.shape)
        xh = self.local_H(xa)
        # print("xh大小", xh.shape)
        xw = self.local_W(xa)
        # print("xw大小", xw.shape)
        xlg =  xl+xh+xw
        wei = self.sigmoid(xlg)
        # print("权重wei", wei.shape)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo
class AFF_CHW4(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels=64, r=4):
        super(AFF_CHW4, self).__init__()
        inter_channels = int(channels // r)


        self.avg_pool_h = nn.AdaptiveAvgPool2d((1, None))
        self.avg_pool_w = nn.AdaptiveAvgPool2d((None, 1))
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )
        self.local_att2 = nn.Sequential(

            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        # print("x",x.shape)
        # print("residual",residual.shape)
        B, C, H, W = x.shape
        xa = x + residual
        xh = self.avg_pool_h(xa)
        xw = self.avg_pool_w(xa)
        xw = xw.permute(0,1, 3, 2)
        xhw = torch.cat([xh, xw], dim=3)
        fhw=self.local_att(xhw)
        # print("fhw", fhw.shape)
        # fhw2 = torch.chunk(fhw, dim=3, chunks=2)
        fhw2 = torch.split(fhw, [H, W], dim = 3)
        x1 = fhw2[0]
        # print("x1", x1.shape)
        x1 = x1.permute(0, 1, 3, 2)
        x2 = fhw2[1]
        # print("x2", x2.shape)
        y1 = self.local_att2(x1)
        y2 = self.local_att2(x2)
        xlg =y1+y2
        wei = self.sigmoid(xlg)
        # print("wei",wei.shape)
        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo
###### IAFF
class iAFF(nn.Module):
    '''
    多特征融合 iAFF
    '''

    def __init__(self, channels=64, r=4):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)

        # 本地注意力
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 第二次本地注意力
        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # 第二次全局注意力
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):  # x一般为低层特征，residual为高层特征
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        # 实现的是什么自适应
        xi = x * wei + residual * (1 - wei)

        xl2 = self.local_att2(xi)
        xg2 = self.global_att(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        return xo


"""
怎么做到自适应融合模块
包括concat模块
和add模块
以及注意机制融合模块
"""


# 第一种自适应融合方式（设计通道注意力方法，让网络决定谁的权重大，进行加权平均）
# -------------------------------------------- #
# 定义SE注意力机制的类
class se_block(nn.Module):
    # 初始化, in_channel代表输入特征图的通道数, ratio代表第一个全连接下降通道的倍数
    def __init__(self, in_channel, ratio=4):
        # 继承父类初始化方法
        super(se_block, self).__init__()

        # 属性分配
        # 全局平均池化，输出的特征图的宽高=1
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        # 第一个全连接层将特征图的通道数下降4倍
        self.fc1 = nn.Linear(in_features=in_channel, out_features=in_channel // ratio, bias=False)
        # relu激活
        self.relu = nn.ReLU()
        # 第二个全连接层恢复通道数
        self.fc2 = nn.Linear(in_features=in_channel // ratio, out_features=in_channel, bias=False)
        # sigmoid激活函数，将权值归一化到0-1
        self.sigmoid = nn.Sigmoid()

    # 前向传播
    def forward(self, inputs):  # inputs 代表输入特征图

        # 获取输入特征图的shape
        b, c, h, w = inputs.shape
        # 全局平均池化 [b,c,h,w]==>[b,c,1,1]
        x = self.avg_pool(inputs)
        # 维度调整 [b,c,1,1]==>[b,c]
        x = x.view([b, c])

        # 第一个全连接下降通道 [b,c]==>[b,c//4]
        x = self.fc1(x)
        x = self.relu(x)
        # 第二个全连接上升通道 [b,c//4]==>[b,c]
        x = self.fc2(x)
        # 对通道权重归一化处理
        x = self.sigmoid(x)
        # print("权重输出x", x.shape)
        # 调整维度 [b,c]==>[b,c,1,1]
        x = x.view([b, c, 1, 1])
        # print("权重输出2x", x)
        # 将输入特征图和通道权重相乘
        # outputs = x * inputs
        return x


# 残差+注意力机制的结果
class SEBottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):  # inplanes, planes输入输出通道数
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 1, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 1)
        self.relu = nn.ReLU(inplace=True)
        # 注意力机制添加位置
        self.se = se_block(planes * 1, reduction)
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
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# #  双模态编码器提取
# class dual_mode_encoder(nn.Module):
#     # Concatenate a list of tensors along dimension
#     def __init__(self, inplanes, outplanes, stride=1, weight2=1):
#         super().__init__()
#         # 最大池化层
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         # 1*1卷积
#         self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False)
#         # BN操作
#         self.bn1 = nn.BatchNorm2d(outplanes)
#         self.conv2 = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=stride,
#                                padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(outplanes)
#         self.conv3 = nn.Conv2d(outplanes, outplanes * 4, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(outplanes * 4)
#
#         # self.w1 = weight1
#         self.w2 = weight2
#     def forward(self,x1,x2):
#         return x1+x2

import matplotlib.pyplot as plt
from torchvision import transforms
import os


# 保存的是每个通道的结果
def feature_visualization2(features, feature_num=100, row=10, col=10):
    """
    features: The feature map which you need to visualization
    model_type: The type of feature map
    model_id: The id of feature map
    feature_num: The amount of visualization you need
    特征：需要可视化的特征图
    model_type：特征图的类型
    model_id：特征图的 id
    feature_num：您需要的可视化量
    """
    # 保存文件的路径
    save_dir = "features/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # print(features.shape)
    # block by channel dimension
    # 打印拆分后的数组大小
    # print(features.shape)
    blocks = torch.chunk(features, features.shape[1], dim=1)
    # print(blocks.shape)
    # # size of feature
    # size = features.shape[2], features.shape[3]

    plt.figure()
    # 每个通道channel绘图大ax上一次
    for i in range(feature_num):
        torch.squeeze(blocks[i])
        #
        feature = transforms.ToPILImage()(blocks[i].squeeze())
        # print(feature)
        ax = plt.subplot(row, col, i + 1)
        # ax = plt.subplot(int(math.sqrt(feature_num)), int(math.sqrt(feature_num)), i+1) #前两个参数m,n决定了将画布分为mn块第三个参数决定了当前选中画布的编号
        ax.set_xticks([])
        ax.set_yticks([])

        plt.imshow(feature)
        # gray feature
        # plt.imshow(feature, cmap='gray')

    # plt.show()
    plt.savefig(save_dir + 'feature_map_2.png', dpi=300)


from torchvision import utils as vutils


# 第一种自适应融合方式（设计通道注意力方法，让网络决定谁的权重大，选择权重大的三层特征层）
class Add_Adaptive(nn.Module):
    def __init__(self, c1, c2, ratio=4, dimension=1):
        super().__init__()
        # self.w1 = weight1
        # self.w2 = weight2
        self.d = dimension
        self.w = se_block(c1, ratio)
        self.c1 = c1

    def forward(self, x1, x2):
        # vutils.save_image(x1[0], 'features/x1.jpg', normalize=True)
        # vutils.save_image(x2[0], 'features/x2.jpg', normalize=True)
        # print("x1",x1)
        # print("x1的大小", x1.shape)
        # print("x2的大小", x2.shape)
        # 先合成6层图像
        x = torch.cat((x1, x2), self.d)
        # print(x.shape)
        # visualize_feature_map(x)
        # 通过六通道图像通过SE通道注意力，保留选择三层权重大的通道
        w_x = self.w(x)
        # print(w_x)
        # print(w_x)
        # sorted, index = torch.sort(w_x[0, :, 0, 0], descending=True)
        # 得到不同通道的权重
        # 将前三个权重叠加，后三个权重叠加
        #
        c1 = self.c1
        output_c = (int)(c1 / 2)
        # feature_visualization2(x,feature_num=100, row=10, col=10)
        w1 = torch.sum(w_x[:, 0:output_c, :, :], 1)
        w2 = torch.sum(w_x[:, output_c:c1, :, :], 1)

        # result_x = torch.add(0.5 * x1, 0.5 * x2)

        wv = torch.div(w1, w1 + w2)
        # 在某个dim部位增加一个维度
        wv = torch.unsqueeze(wv, dim=1)
        # print("权重wv", wv)
        x1 = wv * x1
        wc = torch.div(w2, w1 + w2)
        # print("wv",wv)
        wc = torch.unsqueeze(wc, dim=1)
        # print("权重wc", wc)
        # print("权重w2", wc.shape)
        x2 = wc * x2

        result_x = torch.add(x1, x2)

        # print("各层从大达到小排列", index)
        # for i in w_x[0,:,0,0]:
        #     print("该层权重",i)
        #     print(type(i))
        # print(w_x.shape)

        return result_x


# 注意力机制怎么多层融合，包括先提取特征conv然后残差
# 针对运动物体特征图（使用空间注意力机制）
# class SimAM(torch.nn.Module):
#     def __init__(self, e_lambda=1e-4):
#         super(SimAM, self).__init__()
#
#         self.activaton = nn.Sigmoid()
#         self.e_lambda = e_lambda
#
#     def __repr__(self):
#         s = self.__class__.__name__ + '('
#         s += ('lambda=%f)' % self.e_lambda)
#         return s
#
#     @staticmethod
#     def get_module_name():
#         return "simam"
#
#     def forward(self, x):
#         b, c, h, w = x.size()
#
#         n = w * h - 1
#
#         x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
#         y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
#
#         return x * self.activaton(y)


class Add(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, c1, c2, weight1=1, weight2=1):
        super().__init__()
        self.w1 = weight1
        self.w2 = weight2

    def forward(self, x1, x2):
        # 由于定义了add，则自适应
        # print("w1",self.w1)
        # print("w2",self.w2)
        return torch.add(self.w1 * x1, self.w2 * x2)


# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=4):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         # 一维卷积操作
#         self.f1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
#         self.relu = nn.ReLU()
#         self.f2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
#         # 写法二,亦可使用顺序容器
#         # self.sharedMLP = nn.Sequential(
#         # nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
#         # nn.Conv2d(in_planes // rotio, in_planes, 1, bias=False))
#
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = self.f2(self.relu(self.f1(self.avg_pool(x))))
#         max_out = self.f2(self.relu(self.f1(self.max_pool(x))))
#         out = self.sigmoid(avg_out + max_out)
#         return out
#
#
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#
#         self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv(x)
#         x1 = torch.mean(x)
#         x2 = torch.max(x)
#         x = x1 + x2
#         # print("x1",x1)
#         # print("x2",x2)
#         x = self.sigmoid(x)
#         return x


##################################################################tyu

class DetectMultiBackend(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    # YOLOv5 多后端类，用于在各种后端进行 python 推理
    def __init__(self, weights='yolov5s.pt', device=torch.device('cpu'), dnn=False, data=None, fp16=False, fuse=True):
        # Usage:
        #   PyTorch:              weights = *.pt
        #   TorchScript:                    *.torchscript
        #   ONNX Runtime:                   *.onnx
        #   ONNX OpenCV DNN:                *.onnx --dnn
        #   OpenVINO:                       *_openvino_model
        #   CoreML:                         *.mlmodel
        #   TensorRT:                       *.engine
        #   TensorFlow SavedModel:          *_saved_model
        #   TensorFlow GraphDef:            *.pb
        #   TensorFlow Lite:                *.tflite
        #   TensorFlow Edge TPU:            *_edgetpu.tflite
        #   PaddlePaddle:                   *_paddle_model
        from models.experimental import attempt_download, attempt_load  # scoped to avoid circular import

        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        # 查看权重文件类型pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle, triton
        pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle, triton = self._model_type(w)
        fp16 &= pt or jit or onnx or engine or triton  # FP16
        nhwc = coreml or saved_model or pb or tflite or edgetpu  # BHWC formats (vs torch BCWH)
        stride = 32  # default stride
        cuda = torch.cuda.is_available() and device.type != 'cpu'  # use CUDA
        if not (pt or triton):
            w = attempt_download(w)  # download if not local

        if pt:  # PyTorch
            model = attempt_load(weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=fuse)
            stride = max(int(model.stride.max()), 32)  # model stride
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            model.half() if fp16 else model.float()
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()

        # class names
        if 'names' not in locals():
            names = yaml_load(data)['names'] if data else {i: f'class{i}' for i in range(999)}
        if names[0] == 'n01440764' and len(names) == 1000:  # ImageNet
            names = yaml_load(ROOT / 'data/ImageNet.yaml')['names']  # human-readable names

        self.__dict__.update(locals())  # assign all variables to self

    # 修改
    def forward(self, im, im2, augment=False, visualize=False):
        # YOLOv5 MultiBackend inference
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16
        if self.nhwc:
            im = im.permute(0, 2, 3, 1)  # torch BCHW to numpy BHWC shape(1,320,192,3)

        if self.pt:  # PyTorch
            y = self.model(im, im2, augment=augment, visualize=visualize) if augment or visualize else self.model(im,
                                                                                                                  im2)

            # y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]
            # y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]
            # y[0][..., :4] *= [w, h, w, h]  # xywh normalized to pixels

        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=(1, 3, 640, 640)):
        # Warmup model by running inference once
        warmup_types = self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb, self.triton
        if any(warmup_types) and (self.device.type != 'cpu' or self.triton):
            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            im2 = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            for _ in range(2 if self.jit else 1):  #
                self.forward(im,im2)  # warmup

    @staticmethod
    def _model_type(p='path/to/model.pt'):
        # Return model type from model path, i.e. path='path/to/model.onnx' -> type=onnx
        # types = [pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle]
        from export import export_formats
        from utils.downloads import is_url
        sf = list(export_formats().Suffix)  # export suffixes
        if not is_url(p, check=False):
            check_suffix(p, sf)  # checks
        url = urlparse(p)  # if url may be Triton inference server
        types = [s in Path(p).name for s in sf]
        types[8] &= not types[9]  # tflite &= not edgetpu
        triton = not any(types) and all([any(s in url.scheme for s in ['http', 'grpc']), url.netloc])
        return types + [triton]

    @staticmethod
    def _load_metadata(f=Path('path/to/meta.yaml')):
        # Load metadata from meta.yaml if it exists
        if f.exists():
            d = yaml_load(f)
            return d['stride'], d['names']  # assign stride, names
        return None, None


class AutoShape(nn.Module):
    # YOLOv5 input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    agnostic = False  # NMS class-agnostic
    multi_label = False  # NMS multiple labels per box
    classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
    max_det = 1000  # maximum number of detections per image
    amp = False  # Automatic Mixed Precision (AMP) inference

    def __init__(self, model, verbose=True):
        super().__init__()
        if verbose:
            LOGGER.info('Adding AutoShape... ')
        copy_attr(self, model, include=('yaml', 'nc', 'hyp', 'names', 'stride', 'abc'), exclude=())  # copy attributes
        self.dmb = isinstance(model, DetectMultiBackend)  # DetectMultiBackend() instance
        self.pt = not self.dmb or model.pt  # PyTorch model
        self.model = model.eval()
        if self.pt:
            m = self.model.model.model[-1] if self.dmb else self.model.model[-1]  # Detect()
            m.inplace = False  # Detect.inplace=False for safe multithread inference
            m.export = True  # do not output loss values

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        if self.pt:
            m = self.model.model.model[-1] if self.dmb else self.model.model[-1]  # Detect()
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self

    @smart_inference_mode()
    def forward(self, ims, size=640, augment=False, profile=False):
        # Inference from various sources. For size(height=640, width=1280), RGB images example inputs are:
        #   file:        ims = 'data/images/zidane.jpg'  # str or PosixPath
        #   URI:             = 'https://ultralytics.com/images/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg') or ImageGrab.grab()  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        dt = (Profile(), Profile(), Profile())
        with dt[0]:
            if isinstance(size, int):  # expand
                size = (size, size)
            p = next(self.model.parameters()) if self.pt else torch.empty(1, device=self.model.device)  # param
            autocast = self.amp and (p.device.type != 'cpu')  # Automatic Mixed Precision (AMP) inference
            if isinstance(ims, torch.Tensor):  # torch
                with amp.autocast(autocast):
                    return self.model(ims.to(p.device).type_as(p), augment=augment)  # inference

            # Pre-process
            n, ims = (len(ims), list(ims)) if isinstance(ims, (list, tuple)) else (1, [ims])  # number, list of images
            shape0, shape1, files = [], [], []  # image and inference shapes, filenames
            for i, im in enumerate(ims):
                f = f'image{i}'  # filename
                if isinstance(im, (str, Path)):  # filename or uri
                    im, f = Image.open(requests.get(im, stream=True).raw if str(im).startswith('http') else im), im
                    im = np.asarray(exif_transpose(im))
                elif isinstance(im, Image.Image):  # PIL Image
                    im, f = np.asarray(exif_transpose(im)), getattr(im, 'filename', f) or f
                files.append(Path(f).with_suffix('.jpg').name)
                if im.shape[0] < 5:  # image in CHW
                    im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
                im = im[..., :3] if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)  # enforce 3ch input
                s = im.shape[:2]  # HWC
                shape0.append(s)  # image shape
                g = max(size) / max(s)  # gain
                shape1.append([int(y * g) for y in s])
                ims[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
            shape1 = [make_divisible(x, self.stride) for x in np.array(shape1).max(0)]  # inf shape
            x = [letterbox(im, shape1, auto=False)[0] for im in ims]  # pad
            x = np.ascontiguousarray(np.array(x).transpose((0, 3, 1, 2)))  # stack and BHWC to BCHW
            x = torch.from_numpy(x).to(p.device).type_as(p) / 255  # uint8 to fp16/32

        with amp.autocast(autocast):
            # Inference
            with dt[1]:
                y = self.model(x, augment=augment)  # forward

            # Post-process
            with dt[2]:
                y = non_max_suppression(y if self.dmb else y[0],
                                        self.conf,
                                        self.iou,
                                        self.classes,
                                        self.agnostic,
                                        self.multi_label,
                                        max_det=self.max_det)  # NMS
                for i in range(n):
                    scale_boxes(shape1, y[i][:, :4], shape0[i])

            return Detections(ims, y, files, dt, self.names, x.shape)


class Detections:
    # YOLOv5 detections class for inference results
    def __init__(self, ims, pred, files, times=(0, 0, 0), names=None, shape=None):
        super().__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*(im.shape[i] for i in [1, 0, 1, 0]), 1, 1], device=d) for im in ims]  # normalizations
        self.ims = ims  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.times = times  # profiling times
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple(x.t / self.n * 1E3 for x in times)  # timestamps (ms)
        self.s = tuple(shape)  # inference BCHW shape

    def _run(self, pprint=False, show=False, save=False, crop=False, render=False, labels=True, save_dir=Path('')):
        s, crops = '', []
        for i, (im, pred) in enumerate(zip(self.ims, self.pred)):
            s += f'\nimage {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '  # string
            if pred.shape[0]:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                s = s.rstrip(', ')
                if show or save or render or crop:
                    annotator = Annotator(im, example=str(self.names))
                    for *box, conf, cls in reversed(pred):  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            file = save_dir / 'crops' / self.names[int(cls)] / self.files[i] if save else None
                            crops.append({
                                'box': box,
                                'conf': conf,
                                'cls': cls,
                                'label': label,
                                'im': save_one_box(box, im, file=file, save=save)})
                        else:  # all others
                            annotator.box_label(box, label if labels else '', color=colors(cls))
                    im = annotator.im
            else:
                s += '(no detections)'

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if show:
                if is_jupyter():
                    from IPython.display import display
                    display(im)
                else:
                    im.show(self.files[i])
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                if i == self.n - 1:
                    LOGGER.info(f"Saved {self.n} image{'s' * (self.n > 1)} to {colorstr('bold', save_dir)}")
            if render:
                self.ims[i] = np.asarray(im)
        if pprint:
            s = s.lstrip('\n')
            return f'{s}\nSpeed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {self.s}' % self.t
        if crop:
            if save:
                LOGGER.info(f'Saved results to {save_dir}\n')
            return crops

    @TryExcept('Showing images is not supported in this environment')
    def show(self, labels=True):
        self._run(show=True, labels=labels)  # show results

    def save(self, labels=True, save_dir='runs/detect/exp', exist_ok=False):
        save_dir = increment_path(save_dir, exist_ok, mkdir=True)  # increment save_dir
        self._run(save=True, labels=labels, save_dir=save_dir)  # save results

    def crop(self, save=True, save_dir='runs/detect/exp', exist_ok=False):
        save_dir = increment_path(save_dir, exist_ok, mkdir=True) if save else None
        return self._run(crop=True, save=save, save_dir=save_dir)  # crop results

    def render(self, labels=True):
        self._run(render=True, labels=labels)  # render results
        return self.ims

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        r = range(self.n)  # iterable
        x = [Detections([self.ims[i]], [self.pred[i]], [self.files[i]], self.times, self.names, self.s) for i in r]
        # for d in x:
        #    for k in ['ims', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
        #        setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def print(self):
        LOGGER.info(self.__str__())

    def __len__(self):  # override len(results)
        return self.n

    def __str__(self):  # override print(results)
        return self._run(pprint=True)  # print results

    def __repr__(self):
        return f'YOLOv5 {self.__class__} instance\n' + self.__str__()


class Proto(nn.Module):
    # YOLOv5 mask Proto module for segmentation models
    def __init__(self, c1, c_=256, c2=32):  # ch_in, number of protos, number of masks
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class Classify(nn.Module):
    # YOLOv5 classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self,
                 c1,
                 c2,
                 k=1,
                 s=1,
                 p=None,
                 g=1,
                 dropout_p=0.0):  # ch_in, ch_out, kernel, stride, padding, groups, dropout probability
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, autopad(k, p), g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=dropout_p, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, 1)
        return self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
