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


# 为same卷积或same池化自动扩充
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


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
        # y1=self.conv(x)
        # y1=self.bn(y1)
        # y1=self.act(y1)
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        # print("前",x.shape)
        # y=self.act(self.conv(x))
        # print("后",y.shape)
        return self.act(self.conv(x))


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


class Concat2(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


# 自适应融合模块
class Concat3(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, c1, c2, ratio=16, kernel_size=7, dimension=1):
        super().__init__()
        self.d = dimension  # 沿着哪个维度进行拼接
        self.spatial_attention = SpatialAttention(7)
        self.channel_attention = ChannelAttention(c1, ratio)

    # 注x1为动态图像，x2为红外图像
    def forward(self, x1, x2):
        # 空间注意力生成权重
        weight1 = self.spatial_attention(x1)
        weight2 = self.spatial_attention(x2)
        # print("weight1",weight1)
        # print("weight2",weight2)
        weight = (weight1 / weight2)
        # print("weights",weight)
        # 为什么不能实现复杂场景下的自适应（部分自适应）
        x2 = weight * x2
        x1 = x1 * (2 - weight)
        # 为什么是cat操作而不是add操作（为什么）
        x = torch.cat((x1, x2), self.d)
        # 这里进行修改，X应该是写错了
        # X = self.channel_attention(x)
        # print("输出的大小",x.shape)
        # print("输出的通道注意力权重大小",X.shape)
        X_w = self.channel_attention(x)
        # print("X_W",X_w)
        return x * X_w


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
        print("attn",attn)
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
    def __init__(self, dim, reduction_ratio=32):
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
        # B*C*1*1
        als = self.avgpool(self.als(x))
        # B*C*1*1
        attn = torch.maximum(ac, als)
        # B*1*1*1
        attn = torch.mean(attn, axis=1, keepdim=True)
        # B*1*1*1
        attn = self.softmax(attn)
        # print(attn)
        out = x * attn
        return out


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


# （1）通道注意力机制
class channel_attention(nn.Module):
    # 初始化, in_channel代表输入特征图的通道数, ratio代表第一个全连接的通道下降倍数
    def __init__(self, in_channel, ratio=4):
        # 继承父类初始化方法
        super(channel_attention, self).__init__()

        # 全局最大池化 [b,c,h,w]==>[b,c,1,1]
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=1)
        # 全局平均池化 [b,c,h,w]==>[b,c,1,1]
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

        # 第一个全连接层, 通道数下降4倍
        self.fc1 = nn.Linear(in_features=in_channel, out_features=in_channel // ratio, bias=False)
        # 第二个全连接层, 恢复通道数
        self.fc2 = nn.Linear(in_features=in_channel // ratio, out_features=in_channel, bias=False)

        # relu激活函数
        self.relu = nn.ReLU()
        # sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

    # 前向传播
    def forward(self, inputs):
        # 获取输入特征图的shape
        b, c, h, w = inputs.shape

        # 输入图像做全局最大池化 [b,c,h,w]==>[b,c,1,1]
        max_pool = self.max_pool(inputs)
        # 输入图像的全局平均池化 [b,c,h,w]==>[b,c,1,1]
        avg_pool = self.avg_pool(inputs)

        # 调整池化结果的维度 [b,c,1,1]==>[b,c]
        max_pool = max_pool.view([b, c])
        avg_pool = avg_pool.view([b, c])

        # 第一个全连接层下降通道数 [b,c]==>[b,c//4]
        x_maxpool = self.fc1(max_pool)
        x_avgpool = self.fc1(avg_pool)

        # 激活函数
        x_maxpool = self.relu(x_maxpool)
        x_avgpool = self.relu(x_avgpool)

        # 第二个全连接层恢复通道数 [b,c//4]==>[b,c]
        x_maxpool = self.fc2(x_maxpool)
        x_avgpool = self.fc2(x_avgpool)

        # 将这两种池化结果相加 [b,c]==>[b,c]
        x = x_maxpool + x_avgpool
        # sigmoid函数权值归一化
        x = self.sigmoid(x)
        # print("通道上权重", x.shape)
        # 调整维度 [b,c]==>[b,c,1,1]
        x = x.view([b, c, 1, 1])
        # 输入特征图和通道权重相乘 [b,c,h,w]
        outputs = inputs * x

        return outputs


# ---------------------------------------------------- #
# （2）空间注意力机制
class spatial_attention(nn.Module):
    # 初始化，卷积核大小为7*7
    def __init__(self, kernel_size=7):
        # 继承父类初始化方法
        super(spatial_attention, self).__init__()

        # 为了保持卷积前后的特征图shape相同，卷积时需要padding
        padding = kernel_size // 2
        # 7*7卷积融合通道信息 [b,2,h,w]==>[b,1,h,w]
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size,
                              padding=padding, bias=False)
        # sigmoid函数
        self.sigmoid = nn.Sigmoid()

    # 前向传播
    def forward(self, inputs):
        # 在通道维度上最大池化 [b,1,h,w]  keepdim保留原有深度
        # 返回值是在某维度的最大值和对应的索引
        x_maxpool, _ = torch.max(inputs, dim=1, keepdim=True)

        # 在通道维度上平均池化 [b,1,h,w]
        x_avgpool = torch.mean(inputs, dim=1, keepdim=True)
        # 池化后的结果在通道维度上堆叠 [b,2,h,w]
        x = torch.cat([x_maxpool, x_avgpool], dim=1)

        # 卷积融合通道信息 [b,2,h,w]==>[b,1,h,w]
        x = self.conv(x)
        # 空间权重归一化
        x = self.sigmoid(x)
        # print("空间上权重", x.shape)
        # 输入特征图和空间权重相乘
        outputs = inputs * x

        return outputs


# ---------------------------------------------------- #
# （3）CBAM注意力机制
class cbam(nn.Module):
    # 初始化，in_channel和ratio=4代表通道注意力机制的输入通道数和第一个全连接下降的通道数
    # kernel_size代表空间注意力机制的卷积核大小
    def __init__(self, in_channel, ratio=4, kernel_size=7):
        # 继承父类初始化方法
        super(cbam, self).__init__()

        # 实例化通道注意力机制
        self.channel_attention = channel_attention(in_channel=in_channel, ratio=ratio)
        # 实例化空间注意力机制
        self.spatial_attention = spatial_attention(kernel_size=kernel_size)

    # 前向传播
    def forward(self, inputs):
        # 先将输入图像经过通道注意力机制
        x = self.channel_attention(inputs)
        # 然后经过空间注意力机制
        x = self.spatial_attention(x)
        return x


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
        self.cbam = cbam(planes * 1, ratio=4, kernel_size=7)
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
        # xg = self.avg_pool(xa)
        # xg = self.conv1(xg)
        # xg = self.bn1(xg)
        # xg = self.relu(xg)
        # xg = self.conv2(xg)
        # xg = self.bn2(xg)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
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
        return torch.add(self.w1 * x1, self.w2 * x2)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 一维卷积操作
        self.f1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.f2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        # 写法二,亦可使用顺序容器
        # self.sharedMLP = nn.Sequential(
        # nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
        # nn.Conv2d(in_planes // rotio, in_planes, 1, bias=False))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.f2(self.relu(self.f1(self.avg_pool(x))))
        max_out = self.f2(self.relu(self.f1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        x1 = torch.mean(x)
        x2 = torch.max(x)
        x = x1 + x2
        # print("x1",x1)
        # print("x2",x2)
        x = self.sigmoid(x)
        return x


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
            for _ in range(2 if self.jit else 1):  #
                self.forward(im)  # warmup

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
