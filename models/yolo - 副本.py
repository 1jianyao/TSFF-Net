##########   修改成为自己的双模态的yolov5
""""
主要修改前向通路_forward_once 和 网络构建parse_model
"""

"""
YOLO-specific modules

Usage:
    $ python models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import contextlib
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization, feature_visualization2
from utils.torch_utils import (fuse_conv_and_bn, initialize_weights, model_info, profile, scale_img, select_device,
                               time_sync)

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

"""
Detect模块来搭建Detect层的
将输入的feature map通过一个卷积操作和公式计算想要的shape，为后面的计算损失率或者NMS做准备
"""


class Detect(nn.Module):
    # YOLOv5 Detect head for detection models
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer

        '''
        detection layer 相当于yolov3中的YOLO Layer层
        参数 -- nc: number of classes
             -- anchors: 传入3个feature map上的所有anchor的大小(P3/P4/P5)
             -- ch: [128,256,512] 3个输出feature map的channel
        '''

        super().__init__()
        '''        
                nc:类别数
                no:每个anchor的输出数，为(x,y,w,h,conf) + nc = 5 + nc 的总数
                nl:预测层数，此次为3
                na:anchors的数量，此次为3
                grid:格子坐标系，左上角为(1,1),右下角为(input.w/stride,input.h/stride)
        '''
        # 定义detction layer的一些属性
        self.nc = nc  # number of classes 类别数量coco20为例
        self.no = nc + 5  # number of outputs per anchor 四个坐标信息+目标得分
        self.nl = len(anchors)  # number of detection layers 不同尺度特征图层数
        self.na = len(anchors[0]) // 2  # number of anchors 每个特征图anchors数
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        '''  
                模型中需要保存的参数一般有两种：
                一种是反向传播需要被optimizer更新的，称为parameter;另一种不需要被更新，称为buffer
                buffer的参数更新是在forward中，而optim.step只能更新nn.parameter参数
        '''

        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        # output conv 对每个输出的feature map都要调用一次conv1 x 1
        # 将输出通过卷积到 self.no * self.na 的通道，达到全连接的作用
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        # 一般都是True，默认不使用AWS，Inferentia加速
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    '''
        返回: 一个tensor list，存放三个元素
        [bs, anchor_num, grid_w, grid_h, xywh+c+classes] bs-batch size即多少张图片
        分别是[1,3,80,80,25] [1,3,40,40,25] [1,3,20,20,25]
        inference: 0 [1,19200+4800+1200,25]=[bs,anchor_num*grid_w*grid_h,xywh+c+classes]
    '''

    # 前向传播的最后一层进入这里处理
    # 传入的x为一列表，包含三个元素形状为[batch,128,y,x],[batch,256,y,x],[batch,512,y,x]
    # 相当于一张图最后生成了三组特征向量
    def forward(self, x):
        z = []  # inference output
        # 对3个feature map进行处理
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                '''
                                构造网格
                                因为推理返回的不是归一化后的网络偏移量，需要加上网格的位置，得到最终的推理坐标，再送入NMS
                                所以这里构建网络就是为了记录每个grid的网格坐标，方便后面使用
                                即向前传播时需要将相对坐标转换到grid绝对坐标系中
                                生成的grid坐标系
                                grid[i].shape = [1,1,ny,nx,2]
                                [[[[1,1],[1,2],...[1,nx]],
                                 [[2,1],[2,2],...[2,nx]],
                                     ...,
                                 [[ny,1],[ny,2],...[ny,nx]]]]
                '''

                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                if isinstance(self, Segment):  # (boxes + masks)
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
                else:  # Detect (boxes only)
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, '1.10.0')):
        """构造网格"""
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing='ij') if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


class Segment(Detect):
    # YOLOv5 Segment head for segmentation models
    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), inplace=True):
        super().__init__(nc, anchors, ch, inplace)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.no = 5 + nc + self.nm  # number of outputs per anchor
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.forward

    def forward(self, x):
        p = self.proto(x[0])
        x = self.detect(self, x)
        return (x, p) if self.training else (x[0], p) if self.export else (x[0], p, x[1])


"""
这个模块是整个模型的搭建模块。在yolov5中，该模块不仅包含模型的搭建，还扩展了很多功能，如：特征可视化、打印模型信息、TTA推理增强、
融合Conv + BN加速推理、模型搭载NMS功能、Autoshape函数（模型包含前处理、推理、后处理的模块(预处理 + 推理 + NMS)）。
yolov5的6.0版本包括BaseModel类和DetectionModel类，如下所示。
"""


class BaseModel(nn.Module):
    # YOLOv5 base model
    def forward(self, x1, x2, profile=False, visualize=False):
        return self._forward_once(x1, x2, profile, visualize)  # single-scale inference, train

    '''
        参数 -- x: 输入图像
             -- profile: True 可以做一些性能评估
             -- visualize: True 可以做一些特征可视化
        返回 -- train: 一个tensor，存放三个元素 [bs, anchor_num, grid_w, grid_h, xywh+c+classes]
                        inference: 0 [1,19200+4800+1200,25]=[bs,anchor_num*grid_w*grid_h,xywh+c+classes]
    '''

    # 修改前向推理
    def _forward_once(self, x1, x2, profile=False, visualize=False):
        # y: 存放着self.save=True的每一层的输出，因为后面的层结构Concat等操作要用到
        # dt: 在profile中做性能评估时使用
        y, dt = [], []  # outputs
        # 遍历网络模型(m.i：当前网络层，m.f：当前网络输入来自哪几层)
        for m in self.model:
            # print(m.type)
            # print(x2.shape)
            if m.i < self.fusion_depth:  # backpone1
                if m.type == "models.common.Add":
                    x2 = m(x1, x2)
                elif m.type == "models.common.Add_Adaptive":
                    x2 = m(x1, x2)
                else:
                    x1 = m(x1)
                    x2 = x2
                # 将该层结果x1保存到y中
                y.append(x1 if m.i in self.save else None)  # save output
                # print("经过模型后的大小", x1.shape)
            else:  # backpone2
                # print("x2", type(x2))
                # 如果输出不来自上一层
                if m.f != -1:  # 说明是Concat层
                    if m.type == "models.common.Add_Adaptive":
                        for j in m.f:
                            if j == -1:
                                x2 = x2
                            else:
                                x1 = y[j]
                    if m.type == "models.common.Add":
                        for j in m.f:
                            if j == -1:
                                x2 = x2
                            else:
                                x1 = y[j]
                    elif m.type == "models.common.Concat3":
                        for j in m.f:
                            if j == -1:
                                x2 = x2
                            else:
                                x1 = y[j]
                    elif m.type == "models.common.AFF":
                        for j in m.f:
                            if j == -1:
                                x2 = x2
                            else:
                                x1 = y[j]
                    elif m.type == "models.common.iAFF":
                        for j in m.f:
                            if j == -1:
                                x2 = x2
                            else:
                                x1 = y[j]
                    elif m.type == "models.common.SKConv_dual":
                        for j in m.f:
                            if j == -1:
                                x2 = x2
                            else:
                                x1 = y[j]
                    elif m.type == "models.common.Dual_BA2M":
                        for j in m.f:
                            if j == -1:
                                x2 = x2
                            else:
                                x1 = y[j]
                    elif m.type == "models.common.Multi_BA2M":
                        for j in m.f:
                            if j == -1:
                                x2 = x2
                            else:
                                x1 = y[j]
                    elif m.type in ["models.common.Dual_BA2M_c", "models.common.Dual_BA2M_s",
                                    "models.common.Dual_BA2M_M","models.common.Dual_BA2M_SE",
                                    "models.common.Dual_BA2M_AFF","models.common.Dual_BA2M_AFF_cat","models.common.Dual_BA2M_M_cat"]:
                        for j in m.f:
                            if j == -1:
                                x2 = x2
                            else:
                                x1 = y[j]
                    # 将输入变成指定的那一层(不为-1的那一层)
                    # print(x2.shape)
                    else:
                        # 打包为一个列表
                        x2 = y[m.f] if isinstance(m.f, int) else [x2 if j == -1 else y[j] for j in m.f]

                # 打印日志信息 Flops time等
                if profile:
                    self._profile_one_layer(m, x2, dt)
                # 将输入 输入该层结构中去
                if m.type == "models.common.Add_Adaptive":
                    x2 = m(x1, x2)
                if m.type == "models.common.Add":
                    x2 = m(x1, x2)
                elif m.type == "models.common.Concat3":

                    # x2 = m(x1, x2)
                    x2 = m(x2, x1)  # run
                elif m.type == "models.common.AFF":
                    x2 = m(x1, x2)
                elif m.type == "models.common.iAFF":
                    x2 = m(x1, x2)
                elif m.type == "models.common.SKConv_dual":
                    x2 = m(x1, x2)
                elif m.type == "models.common.Dual_BA2M":
                    x2 = m(x1, x2)
                elif m.type == "models.common.Multi_BA2M":
                    x2 = m(x1, x2)
                elif m.type in ["models.common.Dual_BA2M_c", "models.common.Dual_BA2M_s",
                                "models.common.Dual_BA2M_M","models.common.Dual_BA2M_SE",
                                "models.common.Dual_BA2M_AFF","models.common.Dual_BA2M_AFF_cat",
                                "models.common.Dual_BA2M_M_cat"]:
                    feature_visualization2(x2, m.type, m.i, 1, 1, 1, "x2")
                    feature_visualization2(x1, m.type, m.i,  1, 1, 1,"x1")
                    x2 = m(x1, x2)
                    feature_visualization2(x2, m.type, m.i, 1, 1, 1, "x3")
                else:
                    x2 = m(x2)  # run 正向推理
                    x1 = x1
                # print("当前层",m.i)
                # print(x2.shape)
                # 如果该层为True,保存改成结构输出
                y.append(x2 if m.i in self.save else None)
                # print("经过模型后的大小",x2.shape)
                # 特征可视化
                if visualize:
                    feature_visualization(x2, m.type, m.i, save_dir=visualize)
                # elif m.i==3:
                #     feature_visualization2(x2, m.type, m.i,1,1,1,"dy")
                # elif m.i == 4:
                #     feature_visualization2(x2, m.type, m.i,1,1,1,"fusion")
        # 返回结果
        return x2

    def _profile_one_layer(self, m, x, dt):
        c = m == self.model[-1]  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        """
        用在detect.py、val.py中
        fuse model Conv2d() + BatchNorm2d() layers
        调用torch_utils.py 中 fuse_conv_and_bn 和common.py中的forward_fuse函数
        """
        LOGGER.info('Fusing layers... ')  # 日志
        for m in self.model.modules():  # 遍历每一层结构
            # 如果当前层是卷积层Conv且有BN结构，那么就调用fuse_conv_and_bn函数Conv和BN进行融合，加速推理
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


class DetectionModel(BaseModel):
    # YOLOv5 detection model
    def __init__(self, cfg='yolov5s.yaml', ch=3, ch2=3, nc=None,
                 anchors=None):  # model, input channels, number of classes
        '''
                参数 -- cfg: 模型配置文件
                     -- ch: input img channels 一般是3(RGB文件)
                     -- nc: number of classes 数据集的类别个数
                     -- anchors: 一般是None
                '''

        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml 一般执行这里，因为我们输入参数yaml文件
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            # 如果配置文件中有中文，打开时要加encodeing参数
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        ch2 = self.yaml['ch2'] = self.yaml.get('ch2', ch2)  # input channels

        # 设置类别数 ，不执行，因为nc=self.yaml["nc"]恒成立
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value

        # 重写anchors:一般不执行，因为传进来的anchors一般都是None
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        # 创建网络模型
        # self.model:初始化的整个网络模型（包括Detect层结构）
        # self.save:所有层结构中from不等于-1的序号,并排好序[4 6 10 14 17 20 23],
        # fusion_depth,由于_forward_once需要用到这个参数，所以需要给self赋予这个属性
        self.model, self.save, self.fusion_depth = parse_model(deepcopy(self.yaml), ch=[ch],
                                                               ch2=[ch2])  # model, savelist
        # default class names ["0","1",……]
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        # self.inplace=True 默认True 不使用加速推理
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors
        # 获取Detect模块的stride（相对输入图像的下采样率）
        # 和anchors在当前Detect输出的feature map的尺寸
        # m为yolov5最后输出的三个特征
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            s = 256  # 2x min stride
            # inplace=True
            m.inplace = self.inplace
            # 这一块需要注意（这块是干嘛的）
            # **************************************************************************************************************
            #             forward = lambda x2: self.forward(x2)[0] if isinstance(m, Segment) else self.forward(x2)
            #             # 计算3个feature map的anchor大小,如[10,13]/8 ->[1.25,1.625]
            #             m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            # **************************************************************************************************************
            # x = self.forward(torch.zeros(2, ch, s, s), torch.zeros(2, ch2, s, s))
            m.stride = torch.tensor(
                [s / x.shape[-2] for x in self.forward(torch.zeros(2, ch, s, s), torch.zeros(2, ch2, s, s))])  # forward
            # 检查anchor顺序与stride顺序是否相同
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # only run once

        # Init weights, biases (调用torch_utils.py下initialize_weights初始化模型权重)
        initialize_weights(self)
        self.info()  # 打印模型信息
        LOGGER.info('')

    def forward(self, x1, x2, augment=False, profile=False, visualize=False):
        # 若需要,则要对x1和x2进行相同的操作，进行数据增强则进行上下flip/左右flip
        if augment:
            return self._forward_augment(x1, x2)  # augmented inference, None

        # 默认执行，正常前向推理
        return self._forward_once(x1, x2, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x1, x2):
        img_size = x1.shape[-2:]  # height, width

        s = [1, 0.83, 0.67]  # scales 缩放程度
        f = [None, 3, None]  # flips (2-ud up_down上下flip, 3-lr left-right左右flip)
        y = []  # outputs
        for si, fi in zip(s, f):
            # 缩放图片尺寸（根据f先进行相应翻转）
            xi = scale_img(x1.flip(fi) if fi else x1, si, gs=int(self.stride.max()))
            xi2 = scale_img(x2.flip(fi) if fi else x2, si, gs=int(self.stride.max()))

            yi = self._forward_once(xi, xi2)[0]  # forward

            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            # 将推理结果恢复导相对原图图片尺寸
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)

        '''
        用在上面的_forward_augment函数上
        将推理结果恢复到原图图片尺寸上 TTA中用到
        参数 -- p: 推理结果
             -- flips: 翻转标记(2-ud上下, 3-lr左右)
             -- scale: 图片缩放比例
             -- img_size: 原图图片尺寸
        '''
        # 不同的方式前向推理使用公式不同，具体可看Detect函数

        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency

        ''' 用在上面的__init__函数上 '''
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:5 + m.nc] += math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


# 保留 YOLOv5 “模型”类以实现向后兼容性
Model = DetectionModel  # retain YOLOv5 'Model' class for backwards compatibility


class SegmentationModel(DetectionModel):
    # YOLOv5 segmentation model
    def __init__(self, cfg='yolov5s-seg.yaml', ch=3, nc=None, anchors=None):
        super().__init__(cfg, ch, nc, anchors)


class ClassificationModel(BaseModel):
    # YOLOv5 classification model
    def __init__(self, cfg=None, model=None, nc=1000, cutoff=10):  # yaml, model, number of classes, cutoff index
        super().__init__()
        self._from_detection_model(model, nc, cutoff) if model is not None else self._from_yaml(cfg)

    def _from_detection_model(self, model, nc=1000, cutoff=10):
        # Create a YOLOv5 classification model from a YOLOv5 detection model
        if isinstance(model, DetectMultiBackend):
            model = model.model  # unwrap DetectMultiBackend
        model.model = model.model[:cutoff]  # backbone
        m = model.model[-1]  # last layer
        ch = m.conv.in_channels if hasattr(m, 'conv') else m.cv1.conv.in_channels  # ch into module
        c = Classify(ch, nc)  # Classify()
        c.i, c.f, c.type = m.i, m.f, 'models.common.Classify'  # index, from, type
        model.model[-1] = c  # replace
        self.model = model.model
        self.stride = model.stride
        self.save = []
        self.nc = nc

    def _from_yaml(self, cfg):
        # Create a YOLOv5 classification model from a *.yaml file
        self.model = None


# 用于搭建网络模型，若要修改网络模型框架，则需要对应进行相应的改动
def parse_model(d, ch, ch2):  # model_dict, input_channels(3)
    # Parse a YOLOv5 model.yaml dictionary
    '''
        用在DetectionModel模块中
        解析模型文件(字典形式)，并搭建网络结构
        该函数主要完成：
                更新当前层的args(参数)，计算c2(当前层的输出channel)
                ->使用当前层的参数搭建当前层
                ->生成 layers + save
        参数 -- d: model_dict模型文件，字典形式{dice: 7}(yolov5s.yaml中的6个元素 + ch)
              ch: 记录模型每一层的输出channel，初始ch=[3]，后面会删除
        返回 -- nn.Sequential(*layers): 网络的每一层的层结构
               sorted(save): 把所有层结构中的from不是-1的值记下，并排序[4,6,10,14,17,20,23]
        '''

    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    # 读取指点类型的模型文件参数
    # anchors就是[[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
    # nc类别数，gd网络深度yolov5n=0.33，gw网络宽度yolov5n=0.25,act=None
    anchors, nc, gd, gw, act, fusion_depth = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple'], d.get(
        'activation'), d["fusion_depth"]
    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        LOGGER.info(f"{colorstr('activation:')} {act}")  # print
    # 每一个predict head 上的anchor数(na=3)
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    # 每一个predict head层的输出个数(n0=3*(80+5)))
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    """
    正式搭建网络部分
    layers:保存每一层的层结构
    save：记录下所有层结构中from不是-1的层结构序号
    c2：保存当前层的输出channel
    # c2：新增当前层的输出channel
    """
    layers, save, c2, c22 = [], [], ch[-1], ch2[-1]  # layers, savelist, ch out
    # i:为当前层
    # f(from)：当前层输入来自哪些层
    # n（number）：当前层数(循环次数)
    # m(module):当前层类别
    # args:当前层类别参数列表，包括channel(输入输出通道数)、kernel_size、stride、paddjng和bias
    # 遍历backbone1和backbone2,head的每一层
    for i, (f, n, m, args) in enumerate(d['backbone1'] + d['backbone2'] + d['head']):  # from, number, module, args

        # 得到当前层的真实类名，例如：m=Focus -> <class "models.common Focus">
        m = eval(m) if isinstance(m, str) else m  # eval strings
        # 循环模块参数args(有问题)
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings

                args[j] = eval(a) if isinstance(a, str) else a  # eval strings

        # _________________________更新当前层的args（参数），计算当前层的输出channel
        # gd：depth gain 如yolov5s，n=当前模块的次数
        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        # 当当前层小于backbone1时(c1：当前层的输入channel数；c2:当前层的输出channel数；ch：记录着backbone1的输出channel数，ch2：backbone1+backbone2和head之后的通道数)
        if i < fusion_depth:

            if m in {
                Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x}:
                # c1：当前层的输入channel数；c2:当前层的输出channel数；ch：记录着所有层的输出channel数
                c1, c2 = ch[f], args[0]
                if c2 != no:  # if not output
                    # gw：width gain控制宽度，保证通道数是8的倍数，这样适合GPU训练
                    c2 = make_divisible(c2 * gw, 8)  #
                args = [c1, c2, *args[1:]]
            elif m is Add:  # 使用add融合方式
                # c1 = 3
                # c2 = 3
                c1, c2 = ch[f], 3
                # c2 = 3
                args = args
                # if c2 != no:
                #     c2 = make_divisible(c2 * gw, 8)  # 保证通道是8的倍数
            elif m is Add_Adaptive:
                # 输入通道数
                c1 = ch[f] + ch2[f]
                c2 = c1 / 2
                args = args
            elif m is EIR_cabm:
                # c2 = args[1]
                # c1 = ch[f]
                # args = [c1, *args[0:]]
                c2 = args[1]

                # print("c22",c2)
        # 再backbone2开始和之后
        else:
            if m in {
                Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x}:
                # 需要从第二模态图片重新开始
                c12, c22 = ch2[f], args[0]
                if c22 != no:  # if not output
                    # gw：width gain控制宽度，保证通道数是8的倍数，这样适合GPU训练
                    c22 = make_divisible(c22 * gw, 8)
                # 再初始args的基础上更新，加入当前层的输入channel并更新当前层
                # [in_channels,out_channels,*args[1:]]
                args = [c12, c22, *args[1:]]

                # 如果当前层是BottleneckCSP/ C3/C3TR/C3Ghost/ C3x，则需要在args中加入Bottleneck
                # [in_channels,out_channels,Bottleneck个数，原args[1:]]
                if m in {BottleneckCSP, C3, C3TR, C3Ghost, C3x}:
                    args.insert(2, n)  # number of repeats 在第二位置插入Bottleneck的个数n
                    n = 1  # 恢复默认值
            elif m is nn.BatchNorm2d:
                # BN层只需要返回上一层的输出channel
                args = [ch2[f]]
            # 添加bifpn_concat结构
            elif m in [Concat, BiFPN_Concat2, BiFPN_Concat3]:
                c22 = sum(ch[x] for x in f)
            # 添加bifpn_add结构
            elif m in [BiFPN_Add2, BiFPN_Add3]:
                c2 = max([ch[x] for x in f])
            # Concat2与Concat层结构相同，但是由于最终输出分别存在ch和ch2中
            elif m is Concat2:
                c22 = sum(ch2[x] for x in f)
            elif m is Add:  # 使用add融合方式
                # c1 = 3
                # c2 = 3
                c1, c2 = ch[f[0]], ch[f[0]]
                # c2 = 3
                args = args
            elif m is iAFF:
                c12 = 0
                for x in f:
                    if x == -1:
                        c2m = ch2[x]
                    else:
                        c2m = ch2[x]
                    # 输出通道是两个通道的和
                    c12 = c12 + c2m
                    c22 = int(c12 / 2)
                args = [c22, *args[0:]]
            elif m is AFF:
                c12 = 0
                for x in f:
                    if x == -1:
                        c2m = ch2[x]
                    else:
                        c2m = ch2[x]
                    # 输出通道是两个通道的和
                    c12 = c12 + c2m
                    c22 = int(c12 / 2)
                args = [c22, *args[0:]]
            elif m is Concat3:
                c22 = 0
                for x in f:
                    if x == -1:
                        c2m = ch2[x]
                    else:
                        c2m = ch2[x]
                    c22 = c22 + c2m
                c12 = c22
                args = [c12, c22, *args[0:]]
            elif m is Add_Adaptive:
                # 输入通道数
                # c12 = 2*ch2[f]
                c12 = 0
                for x in f:
                    if x == -1:
                        c2m = ch2[x]
                    else:
                        c2m = ch2[x]
                    c12 = c22 + c2m
                c22 = int(c12 / 2)
                # 这里修改参数
                args = [c12, c22, *args[1:]]
            elif m is EIR_cabm:
                # print("args",args)
                # c1 = ch[f]
                # # args = [c1, *args[0:]]
                # args = args
                c22 = args[1]
            elif m is SKConv:
                c22 = ch2[f]
                args = [c22, c22, *args[0:]]
            elif m is SKConv_dual:
                c22 = ch2[f[0]]
                args = [c22, c22, *args[0:]]
            elif m is Dual_BA2M:
                c12 = ch2[f[0]]
                args = [c12, *args[0:]]
                c22 = c12 * 2
            elif m is Multi_BA2M:
                c12 = ch2[f[0]]
                args = [c12, *args[0:]]
                c22 = c12 * 2
            elif m in [Dual_BA2M_c, Dual_BA2M_s,Dual_BA2M_M,Dual_BA2M_SE,Dual_BA2M_AFF]:
                c12 = ch2[f[0]]
                args = [c12, *args[0:]]
                c22 = c12
            elif m in [Dual_BA2M_AFF_cat,Dual_BA2M_M_cat]:
                c12 = ch2[f[0]]
                args = [c12, *args[0:]]
                c22 = c12*2
            # elif m is AFP:
            #     c22=ch2[f]
            #     args = []
            # TODO: channel, gw, gd
            elif m in {Detect, Segment}:
                # Detect /Segment（Yolo Layer）层
                # 在args中加入3个Detect层的输出channel
                args.append([ch2[x] for x in f])
                if isinstance(args[1], int):  # number of anchors
                    args[1] = [list(range(args[1] * 2))] * len(f)
                if m is Segment:
                    args[3] = make_divisible(args[3] * gw, 8)
            elif m is Contract:
                c22 = ch2[f] * args[0] ** 2
            elif m is Expand:
                c22 = ch2[f] // args[0] ** 2
            else:
                c22 = ch2[f]  # args不变
        # print("当前层结构", m.type)
        # print("当前层参数", args)
        # m_:得到当前层的module,如果n>1就创建多个m（当前层结构），如果n=1就创建一个m（Detect(\
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        # 打印当前层结构的一些基本信息
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        # 计算这一层的参数量
        np = sum(x.numel() for x in m_.parameters())  # number params
        # attach index, 'from' index, type, number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        # 把所有层结构中的from不是-1的值记下来[6 4 14 10 17 20 23]
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        # 将当前层结构module加入layers中
        layers.append(m_)
        # 初始化ch
        if i == 0:
            ch = []  # 去除输入channel[3],刚开始输入
        if i < fusion_depth:
            ch.append(c2)
        # 需要初始化ch2（它应该包含ch的信息）
        if i == fusion_depth:
            ch2 = ch
        ch2.append(c22)
        # # 把当前层的输出channel数加入ch*(这里重复运算，导致结果出错)
        # ch.append(c2)

    # 返回的搭建网络层、以及需要保存的层，
    return nn.Sequential(*layers), sorted(save), fusion_depth


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s_mix_fusion27.yaml', help='model.yaml')
    parser.add_argument('--batch-size', type=int, default=2, help='total batch size for all GPUs')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--line-profile', action='store_true', default=True, help='profile model speed layer by layer')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(vars(opt))

    # # cuda:0
    device = select_device(opt.device)
    # model=Model
    # print(model)
    # 网络先进入DetectionModel，再Parse_model,再进入Detect，
    model = Model(opt.cfg).to(device)
    #
    # # Create model
    # # 可能torch.rand随便找了张图片
    # im1 = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    # im2 = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    # feature = model.forward(im1,im2)
    #
    # # Options
    # if opt.line_profile:  # profile layer by layer
    #     # model(im, profile=True)
    #     pred = model(im1,im2)
    #
    # elif opt.profile:  # profile forward-backward
    #     results = profile(input=im, ops=[model], n=3)
    #
    # elif opt.test:  # test all models
    #     for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
    #         try:
    #             _ = Model(cfg)
    #         except Exception as e:
    #             print(f'Error in {cfg}: {e}')
    #
    # else:  # report fused model summary
    #     model.fuse()
