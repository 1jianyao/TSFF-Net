# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Dataloaders and dataset utils
"""

import contextlib
import glob
import hashlib
import json
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse

import numpy as np
import psutil
import torch
import torch.nn.functional as F
import torchvision
import yaml
from PIL import ExifTags, Image, ImageOps
from torch.utils.data import DataLoader, Dataset, dataloader, distributed
from tqdm import tqdm

from utils.augmentations import (Albumentations, augment_hsv, copy_paste,
                                  letterbox, mixup, random_perspective)
from utils.general import (DATASETS_DIR, LOGGER, NUM_THREADS, TQDM_BAR_FORMAT, check_dataset, check_requirements,
                           check_yaml, clean_str, cv2, is_colab, is_kaggle, segments2boxes, unzip_file, xyn2xy,
                           xywh2xyxy, xywhn2xyxy, xyxy2xywhn)
from utils.torch_utils import torch_distributed_zero_first

# Parameters
HELP_URL = 'See https://docs.ultralytics.com/yolov5/tutorials/train_custom_data'
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
PIN_MEMORY = str(os.getenv('PIN_MEMORY', True)).lower() == 'true'  # global pin_memory for dataloaders
BAR_FORMAT = '{l_bar}{bar:10}{r_bar}{bar:-10b}'  # tqdm bar format
# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def get_hash(paths):
    # Returns a single hash value of a list of paths (files or dirs)
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.sha256(str(size).encode())  # hash sizes
    h.update(''.join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    with contextlib.suppress(Exception):
        rotation = dict(img._getexif().items())[orientation]
        if rotation in [6, 8]:  # rotation 270 or 90
            s = (s[1], s[0])
    return s


def exif_transpose(image):
    """
    Transpose a PIL image accordingly if it has an EXIF Orientation tag.
    Inplace version of https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py exif_transpose()

    :param image: The image to transpose.
    :return: An image.
    """
    exif = image.getexif()
    orientation = exif.get(0x0112, 1)  # default 1
    if orientation > 1:
        method = {
            2: Image.FLIP_LEFT_RIGHT,
            3: Image.ROTATE_180,
            4: Image.FLIP_TOP_BOTTOM,
            5: Image.TRANSPOSE,
            6: Image.ROTATE_270,
            7: Image.TRANSVERSE,
            8: Image.ROTATE_90}.get(orientation)
        if method is not None:
            image = image.transpose(method)
            del exif[0x0112]
            image.info['exif'] = exif.tobytes()
    return image


def seed_worker(worker_id):
    # Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloader(path,
                      path2,
                      imgsz,
                      batch_size,
                      stride,
                      single_cls=False,
                      hyp=None,
                      augment=False,
                      cache=False,
                      pad=0.0,
                      rect=False,
                      rank=-1,
                      workers=8,
                      image_weights=False,
                      quad=False,
                      prefix='',
                      prefix2='',
                      shuffle=False,
                      seed=0):
    """ 参数：
        path: 图片数据加载路径 D:\yolo5-5\yolov5\paper_data\images
        imgsz: train/test图片尺寸（数据增强后大小） 640
        batch_size: batch size 大小 8/16/32
        stride: 模型最大stride=32   [32 16 8]
        single_cls: 数据集是否是单类别 默认False
        hyp: 超参列表dict 网络训练时的一些超参数，包括学习率等，这里主要用到里面一些关于数据增强(旋转、平移等)的系数
        augment: 是否要进行数据增强  True
        cache: 是否cache_images False,不对图片进行缓存
        pad: 设置矩形训练的shape时进行的填充 默认0.0
        rect: 是否开启矩形train/test  默认训练集关闭 验证集开启
        rank:  多卡训练时的进程编号 rank为进程编号  -1且gpu=1时不进行分布式  -1且多块gpu使用DataParallel模式  默认-1
        workers: dataloader的numworks 加载数据时的cpu进程数
        image_weights: 训练时是否根据图片样本真实框分布权重（好像是与数量成反比）来选择图片  默认False
        quad: dataloader取数据时, 是否使用collate_fn4代替collate_fn  默认False
        prefix: 显示信息   一个标志，多为train/val，处理标签时保存cache文件会用到
        """
    # '警告 ⚠️ --rect 与 DataLoader shuffle 不兼容，设置 shuffle=False'
    if rect and shuffle:
        LOGGER.warning('WARNING ⚠️ --rect is incompatible with DataLoader shuffle, setting shuffle=False')
        shuffle = False
    # 双数据集处理
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        # 目标：return torch.from_numpy(img),torch.from_numpy(img2),labels_out，self.im_files[index],self.im_files[index]，shapes
        dataset = LoadImagesAndLabels(
            path, path2,
            imgsz,
            batch_size,
            augment=augment,  # augmentation
            hyp=hyp,  # hyperparameters
            rect=rect,  # rectangular batches
            cache_images=cache,
            single_cls=single_cls,
            stride=int(stride),
            pad=pad,
            image_weights=image_weights,
            prefix=prefix,
            prefix2=prefix2)

    batch_size = min(batch_size, len(dataset))  # 4
    nd = torch.cuda.device_count()  # number of CUDA devices #为-1
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    loader = DataLoader if image_weights else InfiniteDataLoader  # only DataLoader allows for attribute updates
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + seed + RANK)
    return loader(dataset,
                  batch_size=batch_size,
                  shuffle=shuffle and sampler is None,
                  num_workers=nw,
                  sampler=sampler,
                  pin_memory=PIN_MEMORY,
                  collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn,
                  worker_init_fn=seed_worker,
                  generator=generator), dataset


class InfiniteDataLoader(dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)



##------------------------检测图像的时候需要使用（已修改）--------------------------------------
# YOLOv5 图像/视频数据加载器，即“python detect.py --source image.jpg/vid.mp4”
class LoadImages:
    # YOLOv5 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`
    def __init__(self, path, path2, img_size=640, stride=32, auto=True):
        p = str(Path(path).resolve())  # os-agnostic absolute path
        p2 = str(Path(path2).resolve())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
            files2 = sorted(glob.glob(p2, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
            files2 = sorted(glob.glob(os.path.join(p2, '*.*')))  # dir
        elif os.path.isfile(p):  # 如果是文件直接获取
            files = [p]  # files
            files2 = [p2]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')
        # 分别提取图片和视频的路径
        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        images2 = [x for x in files2 if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)  # 获取数量

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos  # 整个图片视频放一个列表
        self.files2 = images2 + videos  # 整个图片视频放一个列表
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv  # 判断是否为视频，方便后续单独处理
        self.mode = 'image'
        self.auto = auto
        if any(videos):  # 是否包含视频文件
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'

    def __iter__(self):  # 创建迭代器对象
        self.count = 0
        return self

    def __next__(self):  # 输出下一项
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]
        path2 = self.files2[self.count]

        if self.video_flag[self.count]:  # 如果为视频
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            s = f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: '

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR格式
            img02 = cv2.imread(path2)  # BGR格式
            assert img0 is not None, f'Image Not Found {path}'
            s = f'image {self.count}/{self.nf} {path}: '

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)[0]  # 对图片缩放填充
        img2 = letterbox(img02, self.img_size, stride=self.stride, auto=self.auto)[0]  # 对图片缩放填充

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB #BGR到RGB的转换
        img = np.ascontiguousarray(img)  # 将数组转换为连续，提高速度
        img2 = img2.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB #BGR到RGB的转换
        img2 = np.ascontiguousarray(img2)  # 将数组转换为连续，提高速度

        return path, path2, img, img0, img2, img02, self.cap, s

    def new_video(self, path):
        self.frame = 0  # frme记录帧数
        self.cap = cv2.VideoCapture(path)  # 初始化视频对象
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 总帧数

    def __len__(self):
        return self.nf  # number of files

# 修改一个适合自己目标检测的类
class LoadImages3:
    # YOLOv5 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`
    def __init__(self, img, img2, img_size=640, stride=32, auto=True, transforms=None, vid_stride=1):
        self.img_size = img_size
        self.stride = stride
        self.files = img
        self.files2 = img2
        self.nf = 1
        self.mode = 'image'
        self.auto = auto
        self.transforms = transforms  # optional
        self.vid_stride = vid_stride  # video frame-rate stride

    # 创造迭代器对象，否则可能无法停止
    def __iter__(self):
        self.count = 0
        return self

    # 输出下一项
    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        img = self.files
        img2 = self.files2
        # Read image
        self.count += 1
        im0 = img  # BGR
        im02 = img2

        # if self.transforms:
        #     im = self.transforms(im0)  # transforms
        #     im2 = self.transforms(im02)  # transforms
        # else:
        # print("处理前",im0.shape)
        im = letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)[0]  # padded resize
        # print("处理后", im0.shape)
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        im2 = letterbox(im02, self.img_size, stride=self.stride, auto=self.auto)[0]  # padded resize
        im2 = im2.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im2 = np.ascontiguousarray(im2)  # contiguous

        return im, im0,im2,im02

    def _cv2_rotate(self, im):
        # Rotate a cv2 video manually
        if self.orientation == 0:
            return cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
        elif self.orientation == 180:
            return cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif self.orientation == 90:
            return cv2.rotate(im, cv2.ROTATE_180)
        return im

    def __len__(self):
        return self.nf  # number of files
# 根据图像路径，找到标签路径（图像1）
def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]

# 根据图像路径，找到标签路径（图像2）
def img2label_paths2(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images2' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]

# 图片标签加载器（图片1和图片2和标签进行打包）
class LoadImagesAndLabels(Dataset):
    # YOLOv5 train_loader/val_loader, loads images and labels for training and validation(为训练加载图像和标签)
    cache_version = 0.6  # dataset labels *.cache version
    rand_interp_methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]

    def __init__(self,
                 path, path2,
                 img_size=640,
                 batch_size=16,
                 augment=False,
                 hyp=None,
                 rect=False,
                 image_weights=False,
                 cache_images=False,
                 single_cls=False,
                 stride=32,
                 pad=0.0,
                 min_items=0,
                 prefix='',
                 prefix2=''):
        # 创建参数
        self.img_size = img_size
        self.augment = augment  # 数据增强
        self.hyp = hyp  # 超参数
        self.image_weights = image_weights  # 图片采集权重
        self.rect = False if image_weights else rect  # 矩形训练
        # mosaic数据增强
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride  # 最大下采样
        self.path = path
        self.path2 = path2
        self.albumentations = Albumentations() if augment else None
        # 一个标志train/val
        self.prefix = prefix
        self.prefix2 = prefix2
        # 找到path中txt文件每一行图像路径im_files
        try:
            f = []  # image files
            # 'D:\\project1\\sf6_dection\\objection_detection4\\yolov5-master\\VOCData\\dataSet_path\\train.txt'
            for p in path if isinstance(path, list) else [path]:
                # 提取文件名
                p = Path(p)  # os-agnostic
                # 如果路径p为文件夹
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('*.*'))  # pathlib
                # 如果路径p为文件
                elif p.is_file():  # file
                    # 读取文件，并自动关闭
                    with open(p) as t:
                        # .read()读取文件，strip()去除首尾空格，.splitlines按每一行读取文件
                        t = t.read().strip().splitlines()
                        # p.parent为上一层的文件夹名+加os.sep=//
                        parent = str(p.parent) + os.sep
                        # f为图像images的相对路径
                        f += [x.replace('./', parent, 1) if x.startswith('./') else x for x in t]  # to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # to global path (pathlib)
                else:
                    raise FileNotFoundError(f'{prefix}{p} does not exist')
            # im_files为对应图像位置.jpg
            self.im_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            # 如果没有找到对应文件，则打印没有No images found
            assert self.im_files, f'{prefix}No images found'
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}\n{HELP_URL}') from e
        # 找到path2中txt文件每一行图像路径im_files2（改变f为f2，p变为p2,t->t2）
        try:
            f2 = []  # image files
            # 'D:\\project1\\sf6_dection\\objection_detection4\\yolov5-master\\VOCData\\dataSet_path\\train.txt'
            for p2 in path2 if isinstance(path2, list) else [path2]:
                # 提取文件名
                p2 = Path(p2)  # os-agnostic
                # 如果路径p为文件夹
                if p2.is_dir():  # dir
                    f2 += glob.glob(str(p2 / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('*.*'))  # pathlib
                # 如果路径p为文件
                elif p2.is_file():  # file
                    # 读取文件，并自动关闭
                    with open(p2) as t2:
                        # .read()读取文件，strip()去除首尾空格，.splitlines按每一行读取文件
                        t2 = t2.read().strip().splitlines()
                        # p.parent为上一层的文件夹名+加os.sep=//
                        parent = str(p2.parent) + os.sep
                        # f2为图像images的相对路径
                        f2 += [x.replace('./', parent, 1) if x.startswith('./') else x for x in t2]  # to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # to global path (pathlib)
                else:
                    raise FileNotFoundError(f'{prefix2}{p2} does not exist')
            # im_files为对应图像位置.jpg
            self.im_files2 = sorted(x.replace('/', os.sep) for x in f2 if x.split('.')[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            # 如果没有找到对应文件，则打印没有No images found
            assert self.im_files2, f'{prefix2}No images found'
        except Exception as e:
            raise Exception(f'{prefix2}Error loading data from {path2}: {e}\n{HELP_URL}') from e

        # Check cache
        # 为对应图像标签标签位置txt(标签只有一个，所以不需要labels2)
        self.label_files = img2label_paths(self.im_files)  # labels
        self.label_files2 = img2label_paths2(self.im_files2)  # labels
        # self.label_files = img2label_paths(self.im_files)  # labels

        # cache_path=D:\project1\sf6_dection\objection_detection4\yolov5-master\VOCData\dataSet_path\train.cache
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')
        cache_path2 = (p2 if p2.is_file() else Path(self.label_files2[0]).parent).with_suffix('.cache')
        # path
        try:
            # 如果有cache文件，直接加载，exist=True；是否已从cache文件中读取nf,nm,ne等信息
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
            # 如果图像版本信息或文件列表的hash值对不上号，可能本地数据集图片和Label可能发生了变化，重新cache label文件
            assert cache['version'] == self.cache_version  # matches current version
            assert cache['hash'] == get_hash(self.label_files + self.im_files)  # identical hash
        except Exception:
            # 否则调用cache_labels缓存标签及标签相关信息
            cache, exists = self.cache_labels(cache_path,self.im_files, self.label_files, prefix), False  # run cache ops
        # path2
        try:
            # 如果有cache文件，直接加载，exist=True；是否已从cache文件中读取nf,nm,ne等信息
            cache2, exists = np.load(cache_path2, allow_pickle=True).item(), True  # load dict
            # 如果图像版本信息或文件列表的hash值对不上号，可能本地数据集图片和Label可能发生了变化，重新cache label文件
            assert cache2['version'] == self.cache_version  # matches current version
            assert cache2['hash'] == get_hash(self.label_files2 + self.im_files2)  # identical hash
        except Exception:
            # 否则调用cache_labels缓存标签及标签相关信息
            cache2, exists = self.cache_labels(cache_path2,self.im_files2, self.label_files2, prefix2), False  # run cache ops

        # Display cache
        # 打印cache的结果nf nm ne nc n=找到的标签数量 漏掉的标签数量 ，空的标签数量 ，损坏的标签数量，总的
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f'Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt'
            # 展示cache result
            tqdm(None, desc=prefix + d, total=n, initial=n, bar_format=TQDM_BAR_FORMAT)  # display cache results
            if cache['msgs']:
                LOGGER.info('\n'.join(cache['msgs']))  # display warnings
        # 数据集没有标签信息，就发出警告并显示标签Label下载help_url
        assert nf > 0 or not augment, f'{prefix}No labels found in {cache_path}, can not start training. {HELP_URL}'

        nf2, nm2, ne2, nc2, n2 = cache2.pop('results')  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f'Scanning {cache_path2}... {nf2} images, {nm2 + ne2} backgrounds, {nc2} corrupt'
            # 展示cache result
            tqdm(None, desc=prefix2 + d, total=n2, initial=n2, bar_format=TQDM_BAR_FORMAT)  # display cache results
            if cache2['msgs']:
                LOGGER.info('\n'.join(cache2['msgs']))  # display warnings
        # 数据集没有标签信息，就发出警告并显示标签Label下载help_url
        assert nf2 > 0 or not augment, f'{prefix2}No labels found in {cache_path2}, can not start training. {HELP_URL}'

        # Read cache
        # 先从cache中去除cache文件中其它无关键值如：“hash”，“verion”,"mags"等
        [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
        # 只剩下cache[img_file]=[l, shape, segments]

        # cache.values(): 取cache中所有值 对应所有l, shape, segments
        # labels: 如果数据集所有图片中没有一个多边形label  labels存储的label就都是原始label(都是正常的矩形label)
        #         否则将所有图片正常gt的label存入labels 不正常gt(存在一个多边形)经过segments2boxes转换为正常的矩形label
        # shapes: 所有图片的shape
        # self.segments: 如果数据集所有图片中没有一个多边形label  self.segments=None
        #                否则存储数据集中所有存在多边形gt的图片的所有原始label(肯定有多边形label 也可能有矩形正常label 未知数)
        # zip 是因为cache中所有labels、shapes、segments信息都是按每张img分开存储的, zip是将所有图片对应的信息叠在一起

        labels, shapes, self.segments = zip(*cache.values())
        # 框的总个数
        nl = len(np.concatenate(labels, 0))  # number of labels
        assert nl > 0 or not augment, f'{prefix}All labels empty in {cache_path}, can not start training. {HELP_URL}'
        self.labels = list(labels)
        self.shapes = np.array(shapes)
        # 更新所有图片的img_files信息
        self.im_files = list(cache.keys())  # update 图片列表
        # 更新所有图片的label_files信息
        self.label_files = img2label_paths(cache.keys())  # update 标签列表

        # Read cache2
        # 先从cache中去除cache文件中其它无关键值如：“hash”，“verion”,"mags"等
        [cache2.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
        # [cache2.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
        # 只剩下cache[img_file]=[l, shape, segments]

        # cache.values(): 取cache中所有值 对应所有l, shape, segments
        # labels: 如果数据集所有图片中没有一个多边形label  labels存储的label就都是原始label(都是正常的矩形label)
        #         否则将所有图片正常gt的label存入labels 不正常gt(存在一个多边形)经过segments2boxes转换为正常的矩形label
        # shapes: 所有图片的shape
        # self.segments: 如果数据集所有图片中没有一个多边形label  self.segments=None
        #                否则存储数据集中所有存在多边形gt的图片的所有原始label(肯定有多边形label 也可能有矩形正常label 未知数)
        # zip 是因为cache中所有labels、shapes、segments信息都是按每张img分开存储的, zip是将所有图片对应的信息叠在一起

        labels2, shapes2, self.segments2 = zip(*cache2.values())

        self.labels2 = list(labels2)
        self.shapes2 = np.array(shapes2)
        # 更新所有图片的img_files信息
        self.im_files2 = list(cache2.keys())  # update 图片列表
        # 更新所有图片的label_files信息
        self.label_files2 = img2label_paths2(cache2.keys())  # update 标签列表
        n = len(self.shapes)  # number of images
        # 图像的数量/batch_size
        bi = np.floor(np.arange(n) / batch_size).astype(int)  # batch index
        # 以这样batch_size,一轮训练需要循环的次数
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)
        # Create indices
        # 图像的数量
        n2 = len(self.shapes2)  # number of images
        # 图像的数量/batch_size
        bi2 = np.floor(np.arange(n2) / batch_size).astype(int)  # batch index
        # 以这样batch_size,一轮训练需要循环的次数
        nb2 = bi2[-1] + 1  # number of batches
        self.batch2 = bi2  # batch index of image
        self.n2 = n2
        self.indices2 = range(n2)

        # Update labels
        include_class = []  # filter labels to include only these classes (optional)
        self.segments = list(self.segments)
        include_class_array = np.array(include_class).reshape(1, -1)
        for i, (label, segment) in enumerate(zip(self.labels, self.segments)):
            if include_class:
                j = (label[:, 0:1] == include_class_array).any(1)
                self.labels[i] = label[j]
                if segment:
                    self.segments[i] = [segment[idx] for idx, elem in enumerate(j) if elem]
            if single_cls:  # single-class training, merge all classes into 0
                self.labels[i][:, 0] = 0

        # Update labels2
        include_class2 = []  # filter labels to include only these classes (optional)
        self.segments2 = list(self.segments2)
        include_class_array2 = np.array(include_class2).reshape(1, -1)
        # 取出标签（按顺序取出标签）
        for i, (label2, segment2) in enumerate(zip(self.labels2, self.segments2)):
            if include_class2:
                j = (label2[:, 0:1] == include_class_array2).any(1)
                self.labels2[i] = label2[j]
                if segment2:
                    self.segments2[i] = [segment2[idx] for idx, elem in enumerate(j) if elem]
            if single_cls:  # single-class training, merge all classes into 0
                self.labels2[i][:, 0] = 0

        # Rectangular Training（矩形训练，这边好像没有使用）
        # 主要注意shapes的生成，这一步很重要，因为如果采样矩阵训练，那么整个batch要一样
        if self.rect:

            ######## 第一部分
            # Sort by aspect ratio
            # 排序
            s = self.shapes  # wh宽高
            ar = s[:, 1] / s[:, 0]  # aspect ratio宽高比
            irect = ar.argsort()  # 排序
            self.im_files = [self.im_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.segments = [self.segments[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # 计算每个batch采用统一尺寸
            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]
            # 计算要求每个batch输入网络的shape值（向上设置为32的整数倍）
            # 要求每个batch_shapes的高宽都是32的整数倍，所以要先除以32，再取整再乘以32
            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(int) * stride

            ####### 第二部分
            s2 = self.shapes2  # wh宽高
            ar2 = s2[:, 1] / s2[:, 0]  # aspect ratio宽高比
            irect2 = ar2.argsort()  # 排序
            self.im_files2 = [self.im_files2[i] for i in irect2]
            self.label_files2 = [self.label_files2[i] for i in irect2]
            self.labels2 = [self.labels2[i] for i in irect2]
            self.segments2 = [self.segments2[i] for i in irect2]
            self.shapes2 = s[irect2]  # wh
            ar2 = ar2[irect2]

            # 计算每个batch采用统一尺寸
            # Set training image shapes
            shapes2 = [[1, 1]] * nb2
            for i in range(nb2):
                ari2 = ar2[bi2 == i]
                mini, maxi = ari2.min(), ari2.max()
                if maxi < 1:
                    shapes2[i] = [maxi, 1]
                elif mini > 1:
                    shapes2[i] = [1, 1 / mini]
            # 计算要求每个batch输入网络的shape值（向上设置为32的整数倍）
            # 要求每个batch_shapes的高宽都是32的整数倍，所以要先除以32，再取整再乘以32
            self.batch_shapes2 = np.ceil(np.array(shapes2) * img_size / stride + pad).astype(int) * stride

        # Cache images into RAM/disk for faster training
        # 这边好像也没有使用
        if cache_images == 'ram' and not self.check_cache_ram(prefix=prefix):
            cache_images = False
        # 将图片加载到内存中，此处使用了多线程加载图片，可以很快的提高速度
        # 注：在
        self.ims = [None] * n
        self.ims2 = [None] * n2
        # 图像转成对应的npy
        self.npy_files = [Path(f).with_suffix('.npy') for f in self.im_files]
        self.npy_files2 = [Path(f).with_suffix('.npy') for f in self.im_files2]

        # 这边也没执行(cache_images)，不对图像进行缓存

    # # 好像没有被使用
    # def check_cache_ram(self, safety_margin=0.1, prefix=''):
    #     # Check image caching requirements vs available memory
    #     b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
    #     n = min(self.n, 30)  # extrapolate from 30 random images
    #     for _ in range(n):
    #         im = cv2.imread(random.choice(self.im_files))  # sample image
    #         ratio = self.img_size / max(im.shape[0], im.shape[1])  # max(h, w)  # ratio
    #         b += im.nbytes * ratio ** 2
    #     mem_required = b * self.n / n  # GB required to cache dataset into RAM
    #     mem = psutil.virtual_memory()
    #     cache = mem_required * (1 + safety_margin) < mem.available  # to cache or not to cache, that is the question
    #     if not cache:
    #         LOGGER.info(f'{prefix}{mem_required / gb:.1f}GB RAM required, '
    #                     f'{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, '
    #                     f"{'caching images ✅' if cache else 'not caching images ⚠️'}")
    #     return cache

    # 在上面的_init_函数调用，用于缓存标签及标签相关信息
    # def cache_labels(self, path=Path('./labels.cache'), prefix=''):
    #     # Cache dataset labels, check images and read shapes
    #     x = {}  # dict
    #     nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
    #     desc = f'{prefix}Scanning {path.parent / path.stem}...'
    #     with Pool(NUM_THREADS) as pool:
    #         pbar = tqdm(pool.imap(verify_image_label, zip(self.im_files, self.label_files, repeat(prefix))),
    #                     desc=desc,
    #                     total=len(self.im_files),
    #                     bar_format=TQDM_BAR_FORMAT)
    #         for im_file, lb, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
    #             nm += nm_f
    #             nf += nf_f
    #             ne += ne_f
    #             nc += nc_f
    #             if im_file:
    #                 x[im_file] = [lb, shape, segments]
    #             if msg:
    #                 msgs.append(msg)
    #             pbar.desc = f'{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt'
    #
    #     pbar.close()
    #     if msgs:
    #         LOGGER.info('\n'.join(msgs))
    #     if nf == 0:
    #         LOGGER.warning(f'{prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}')
    #     x['hash'] = get_hash(self.label_files + self.im_files)
    #     x['results'] = nf, nm, ne, nc, len(self.im_files)
    #     x['msgs'] = msgs  # warnings
    #     x['version'] = self.cache_version  # cache version
    #     try:
    #         np.save(path, x)  # save cache for next time
    #         path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
    #         LOGGER.info(f'{prefix}New cache created: {path}')
    #     except Exception as e:
    #         LOGGER.warning(f'{prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable: {e}')  # not writeable
    #     return x
    # 生成cache文件（已修改）
    def cache_labels(self, path, im_files, label_files, prefix=''):
        cache_version = 0.6
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels..."
        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(pool.imap(verify_image_label, zip(im_files, label_files, repeat(prefix))),desc=desc, total=len(im_files), bar_format=BAR_FORMAT)
            for im_file, lb, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x[im_file] = [lb, shape, segments]  # 保存为字典
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc}{nf} found, {nm} missing, {ne} empty, {nc} corrupt"

        pbar.close()
        if msgs:
            LOGGER.info('\n'.join(msgs))
        if nf == 0:
            LOGGER.warning(f'{prefix}WARNING: No labels found in {path}. See {HELP_URL}')
        x['hash'] = get_hash(label_files + im_files)
        x['results'] = nf, nm, ne, nc, len(im_files)
        x['msgs'] = msgs  # warnings
        x['version'] = cache_version  # cache version
        try:
            np.save(path, x)  # save cache for next time 保存本地方便下次使用
            path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
            LOGGER.info(f'{prefix}New cache created: {path}')
        except Exception as e:
            LOGGER.warning(f'{prefix}WARNING: Cache directory {path.parent} is not writeable: {e}')  # not writeable
        return x

    # 实际输入图像大小和输入模型的图像需要进行resize操作
    # # 加载图片并根据设定输入大小与图片源大小比例进行resize
    def __len__(self):
        return len(self.im_files)

    #
    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    # 数据增强（图像增强，但图像1、和图像2需要进行相同的数据增强，否则融合后位置出错，标签文件也需要进行相同的位置修改）
    def __getitem__(self, index):
        # 图片索引index
        i = self.indices[index]  # linear, shuffled, or image_weights
        i2 = self.indices2[index]  # linear, shuffled, or image_weights
        # 超参数 包含很多数据增强超参数
        hyp = self.hyp
        # 是否进行马赛克数据增强
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        if mosaic:
            # Load mosaic（被调用）
            img, img2, labels = self.load_mosaic(index)
            shapes = None
            shapes2 = None
            # 查看马赛克结果图
            # cv2.imshow("masaic", img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # # MixUp augmentation（混合图像数据增强）
            # if random.random() < hyp['mixup']:
            #     # img:   两张图片融合之后的图片--numpy (640, 640, 3)
            #     # labels: 两张图片融合之后的标签--label [M+N, cls+x1y1x2y2]
            #     img, labels = mixup(img, labels, *self.load_mosaic(random.randint(0, self.n - 1)))
        # 不进行马赛克
        else:
            # Load image（载入图像）
            # 1、img: resize后的图片； 2、(h0, w0): 原始图片的hw； 3、(h, w): resize后的图片的hw
            img, (h0, w0), (h, w) = self.load_image(index)
            img2, (h02, w02), (h2, w2) = self.load_image2(index)
            # Letterbox
            # 4、Letterbox 主要运用于val或者detect
            # 4.1 确定这张当前图片letterbox之后的shape
            # 如果不用self.rect矩形训练shape就是self.img_size
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            # 4.2 letterbox 这一步将第一步缩放得到的图片再缩放到当前batch所需要的尺度
            # 矩形推理需要一个batch的所有图片的shape必须相同
            # 这里没有缩放操作，所以这里的ratio永远都是(1.0, 1.0)  pad=(0.0, 20.5)
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling
            # 图片letterbox之后label的坐标也要相应变化  根据pad调整label坐标 并将归一化的xywh -> 未归一化的xyxy
            labels = self.labels[index].copy()
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            shape2 = self.batch_shapes2[self.batch2[index]] if self.rect else self.img_size  # final letterboxed shape
            # 4.2 letterbox 这一步将第一步缩放得到的图片再缩放到当前batch所需要的尺度
            # 矩形推理需要一个batch的所有图片的shape必须相同
            # 这里没有缩放操作，所以这里的ratio永远都是(1.0, 1.0)  pad=(0.0, 20.5)
            img2, ratio2, pad2 = letterbox(img2, shape2, auto=False, scaleup=self.augment)
            shapes2 = (h02, w02), ((h2 / h02, w2 / w02), pad2)  # for COCO mAP rescaling
            # 图片letterbox之后label的坐标也要相应变化  根据pad调整label坐标 并将归一化的xywh -> 未归一化的xyxy
            labels2 = self.labels2[index].copy()
            if labels2.size:  # normalized xywh to pixel xyxy format
                labels2[:, 1:] = xywhn2xyxy(labels2[:, 1:], ratio2[0] * w2, ratio2[1] * h2, padw=pad2[0], padh=pad2[1])

            if self.augment:
                # 5、random_perspective增强: 随机对图片进行旋转，平移，缩放，裁剪，透视变换
                img, img2, labels = random_perspective(img, img2,
                                                       labels,
                                                       degrees=hyp['degrees'],
                                                       translate=hyp['translate'],
                                                       scale=hyp['scale'],
                                                       shear=hyp['shear'],
                                                       perspective=hyp['perspective'])
        # labels的数量
        nl = len(labels)  # number of labels
        if nl:  # 若不为0
            # 转换Label尺寸个数 将未归一化的xyxy ->归一化的xywz
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)

        if self.augment:
            # Albumentations
            img, img2, labels = self.albumentations(img, img2, labels)
            nl = len(labels)  # update after albumentations

            # HSV color-space（hsv色域空间增强）
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])
            augment_hsv(img2, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])
            # Flip up-down（随机上旋翻转）
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                img2 = np.flipud(img2)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            # Flip left-right（随机左右翻转）
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                img2 = np.fliplr(img2)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

            # Cutouts
            # labels = cutout(img, labels, p=0.5)
            # nl = len(labels)  # update after cutout
        # 初始化标签框对应的图片序号，配合下面的collate_fn使用
        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB（图像BGR转变成RGB）
        img2 = img2.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)  # 加快运算
        img2 = np.ascontiguousarray(img2)  # 加快运算
        return torch.from_numpy(img), torch.from_numpy(img2), labels_out, self.im_files[index], self.im_files2[
            index], shapes, shapes2

    """   
    返回值：
    1、torch.from_numpy(img): 这个index的图片数据(增强后) [3, 640, 640]
    2、labels_out: 这个index图片的gt label [6, 6] = [gt_num, 0+class+xywh(normalized)]
    3、self.img_files[index]: 这个index图片的路径地址
    4、shapes: 这个batch的图片的shapes 测试时(矩形训练)才有  验证时为None
    """

    # 已修改
    def load_image(self, i):
        # Loads 1 image from dataset index 'i', returns (im, original hw, resized hw)
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i],
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                im = np.load(fn)
            else:  # read image
                im = cv2.imread(f)  # BGR
                assert im is not None, f'Image Not Found {f}'
            h0, w0 = im.shape[:2]  # orig hw
            r = self.img_size / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
                im = cv2.resize(im, (math.ceil(w0 * r), math.ceil(h0 * r)), interpolation=interp)
            return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
        return self.ims[i], self.im_hw0[i], self.im_hw[i]  # im, hw_original, hw_resized

    """返回为图像，原始图像hw，resize之后图像hw"""

    # 已修改（主要修改图像、图像路径、npy图像路径）如果没有该图像，则打印该图像文件没有被发现
    def load_image2(self, i):
        # Loads 1 image from dataset index 'i', returns (im, original hw, resized hw)
        im, f, fn = self.ims2[i], self.im_files2[i], self.npy_files2[i],
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                im = np.load(fn)
            else:  # read image
                im = cv2.imread(f)  # BGR
                assert im is not None, f'Image Not Found {f}'
            h0, w0 = im.shape[:2]  # orig hw
            r = self.img_size / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
                im = cv2.resize(im, (math.ceil(w0 * r), math.ceil(h0 * r)), interpolation=interp)
            return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
        return self.ims[i], self.im_hw0[i], self.im_hw[i]  # im, hw_original, hw_resized

    # """返回为图像，原始图像hw，resize之后图像hw"""
    # def cache_images_to_disk(self, i):
    #     # Saves an image as an *.npy file for faster loading
    #     # 将图像另存为 *.npy 文件以加快加载速度
    #     f = self.npy_files[i]
    #     if not f.exists():
    #         np.save(f.as_posix(), cv2.imread(self.im_files[i]))

    # 已修改（拼接、马赛克等等数据增强）
    def load_mosaic(self, index):  # self自定义数据集 index要增强的索引
        # YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
        labels4, segments4 = [], []
        s = self.img_size
        # 随机选取一个中心点
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y
        indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
        # 随机取其他三张图片索引
        random.shuffle(indices)
        for i, index in enumerate(indices):
            # Load image
            img, _, (h, w) = self.load_image(index)  # load_image 加载图片根据设定的输入大小与图片原大小的比例进行resize
            img2, _, (h2, w2) = self.load_image2(index)  # load_image 加载图片根据设定的输入大小与图片原大小的比例进行resize

            # place img in img4
            if i == 0:  # top left
                # 初始化大图
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                img42 = np.full((s * 2, s * 2, img2.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                # 把原图放到左上角
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                # 选取小图上的位置 如果图片越界会裁剪
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            # 小图上截取的部分贴到大图上
            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            img42[y1a:y2a, x1a:x2a] = img2[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            # 计算小图到大图后的偏移 用来确定目标框的位置
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            labels, segments = self.labels[index].copy(), self.segments[index].copy()
            # 标签裁剪
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
                segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
            labels4.append(labels)  # 得到新的label的坐标
            segments4.extend(segments)

        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        for x in (labels4[:, 1:], *segments4):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        # img4, labels4 = replicate(img4, labels4)  # replicate

        # Augment
        # 将图片中没目标的 取别的图进行粘贴
        img4, img42, labels4, segments4 = copy_paste(img4, img42, labels4, segments4, p=self.hyp['copy_paste'])
        # 随机变换
        img4, img42, labels4 = random_perspective(img4, img42, labels4, segments4,
                                                  degrees=self.hyp['degrees'],
                                                  translate=self.hyp['translate'],
                                                  scale=self.hyp['scale'],
                                                  shear=self.hyp['shear'],
                                                  perspective=self.hyp['perspective'],
                                                  border=self.mosaic_border)  # border to remove

        return img4, img42, labels4  # 返回数据增强的后的图片和标签

    # def load_mosaic(self, index):
    #     # YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
    #     # YOLOv5 4 马赛克加载器。将 1 张图像 + 3 张随机图像加载到 4 张图像的马赛克中
    #     labels4, segments4 = [], []
    #     s = self.img_size
    #     yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y
    #     indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
    #     random.shuffle(indices)
    #     for i, index in enumerate(indices):
    #         # Load image
    #         img, _, (h, w) = self.load_image(index)
    #
    #         # place img in img4
    #         if i == 0:  # top left
    #             img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
    #             x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
    #             x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
    #         elif i == 1:  # top right
    #             x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
    #             x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
    #         elif i == 2:  # bottom left
    #             x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
    #             x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
    #         elif i == 3:  # bottom right
    #             x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
    #             x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
    #
    #         img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
    #         padw = x1a - x1b
    #         padh = y1a - y1b
    #
    #         # Labels
    #         labels, segments = self.labels[index].copy(), self.segments[index].copy()
    #         if labels.size:
    #             labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
    #             segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
    #         labels4.append(labels)
    #         segments4.extend(segments)
    #
    #     # Concat/clip labels
    #     labels4 = np.concatenate(labels4, 0)
    #     for x in (labels4[:, 1:], *segments4):
    #         np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    #     # img4, labels4 = replicate(img4, labels4)  # replicate
    #
    #     # Augment
    #     img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp['copy_paste'])
    #     img4, labels4 = random_perspective(img4,
    #                                        labels4,
    #                                        segments4,
    #                                        degrees=self.hyp['degrees'],
    #                                        translate=self.hyp['translate'],
    #                                        scale=self.hyp['scale'],
    #                                        shear=self.hyp['shear'],
    #                                        perspective=self.hyp['perspective'],
    #                                        border=self.mosaic_border)  # border to remove
    #
    #     return img4, labels4

    # def load_mosaic9(self, index):
    #     # YOLOv5 9-mosaic loader. Loads 1 image + 8 random images into a 9-image mosaic
    #     # YOLOv5 9 马赛克加载器。将 1 张图像 + 8 张随机图像加载到 9 张图像的马赛克中
    #     labels9, segments9 = [], []
    #     s = self.img_size
    #     indices = [index] + random.choices(self.indices, k=8)  # 8 additional image indices
    #     random.shuffle(indices)
    #     hp, wp = -1, -1  # height, width previous
    #     for i, index in enumerate(indices):
    #         # Load image
    #         img, _, (h, w) = self.load_image(index)
    #
    #         # place img in img9
    #         if i == 0:  # center
    #             img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
    #             h0, w0 = h, w
    #             c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
    #         elif i == 1:  # top
    #             c = s, s - h, s + w, s
    #         elif i == 2:  # top right
    #             c = s + wp, s - h, s + wp + w, s
    #         elif i == 3:  # right
    #             c = s + w0, s, s + w0 + w, s + h
    #         elif i == 4:  # bottom right
    #             c = s + w0, s + hp, s + w0 + w, s + hp + h
    #         elif i == 5:  # bottom
    #             c = s + w0 - w, s + h0, s + w0, s + h0 + h
    #         elif i == 6:  # bottom left
    #             c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
    #         elif i == 7:  # left
    #             c = s - w, s + h0 - h, s, s + h0
    #         elif i == 8:  # top left
    #             c = s - w, s + h0 - hp - h, s, s + h0 - hp
    #
    #         padx, pady = c[:2]
    #         x1, y1, x2, y2 = (max(x, 0) for x in c)  # allocate coords
    #
    #         # Labels
    #         labels, segments = self.labels[index].copy(), self.segments[index].copy()
    #         if labels.size:
    #             labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padx, pady)  # normalized xywh to pixel xyxy format
    #             segments = [xyn2xy(x, w, h, padx, pady) for x in segments]
    #         labels9.append(labels)
    #         segments9.extend(segments)
    #
    #         # Image
    #         img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]  # img9[ymin:ymax, xmin:xmax]
    #         hp, wp = h, w  # height, width previous
    #
    #     # Offset
    #     yc, xc = (int(random.uniform(0, s)) for _ in self.mosaic_border)  # mosaic center x, y
    #     img9 = img9[yc:yc + 2 * s, xc:xc + 2 * s]
    #
    #     # Concat/clip labels
    #     labels9 = np.concatenate(labels9, 0)
    #     labels9[:, [1, 3]] -= xc
    #     labels9[:, [2, 4]] -= yc
    #     c = np.array([xc, yc])  # centers
    #     segments9 = [x - c for x in segments9]
    #
    #     for x in (labels9[:, 1:], *segments9):
    #         np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    #     # img9, labels9 = replicate(img9, labels9)  # replicate
    #
    #     # Augment
    #     img9, labels9, segments9 = copy_paste(img9, labels9, segments9, p=self.hyp['copy_paste'])
    #     img9, labels9 = random_perspective(img9,
    #                                        labels9,
    #                                        segments9,
    #                                        degrees=self.hyp['degrees'],
    #                                        translate=self.hyp['translate'],
    #                                        scale=self.hyp['scale'],
    #                                        shear=self.hyp['shear'],
    #                                        perspective=self.hyp['perspective'],
    #                                        border=self.mosaic_border)  # border to remove
    #
    #     return img9, labels9
    # 已修改
    @staticmethod
    def collate_fn(batch):  # 如何取样本
        """
        整理函数  将image和label整合到一起
        pytorch的DataLoader打包一个batch的数据集时要经过此函数进行打包 通过重写此函数实现标签与图片对应的划分，
        一个batch中哪些标签属于哪一张图片,形如
            [[0, 6, 0.5, 0.5, 0.26, 0.35],
             [0, 6, 0.5, 0.5, 0.26, 0.35],
             [1, 6, 0.5, 0.5, 0.26, 0.35],
             [2, 6, 0.5, 0.5, 0.26, 0.35],]
             图片编号、labels类别编号 xywh
        """
        # img: 一个tuple 由batch_size个tensor组成 整个batch中每个tensor表示一张图片
        # label: 一个tuple 由batch_size个tensor组成 每个tensor存放一张图片的所有的target信息
        # label[6, object_num] 6中的第一个数代表一个batch中的第几张图
        # path: 一个tuple 由4个str组成, 每个str对应一张图片的地址信息

        im, im2, label, path, path2, shapes, shape2 = zip(*batch)  # transposed
        for i, lb in enumerate(label):
            lb[:, 0] = i  # add target image index for build_targets()
        # torch.stack(img, 0): 整个batch的图片--numpy
        # torch.cat(label, 0): [num_target, img_index+class_index+xywh(normalized)] 整个batch的label
        # path: 整个batch所有图片的路径
        # shapes: (h0, w0), ((h / h0, w / w0), pad)
        return torch.stack(im, 0), torch.stack(im2, 0), torch.cat(label, 0), path, path2, shapes, shape2

    # 已修改
    @staticmethod
    def collate_fn4(batch):
        """同样在create_dataloader中生成dataloader时调用：
        这里是yolo-v5作者实验性的一个代码 quad-collate function 当train.py的opt参数quad=True 则调用collate_fn4代替collate_fn
        作用:  如之前用collate_fn可以返回图片[16, 3, 640, 640] 经过collate_fn4则返回图片[4, 3, 1280, 1280]
              将4张mosaic图片[1, 3, 640, 640]合成一张大的mosaic图片[1, 3, 1280, 1280]
              将一个batch的图片每四张处理, 0.5的概率将四张图片拼接到一张大图上训练, 0.5概率直接将某张图片上采样两倍训练
        """
        im, im2, label, path, path2, shapes, shapes2 = zip(*batch)  # transposed
        n = len(shapes) // 4
        im4, im42, label4, path4, path42, shapes4, shapes42 = [], [], [], path[:n], path[:n2], shapes[:n], shapes[:n2]

        ho = torch.tensor([[0.0, 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0.0, 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, 0.5, 0.5, 0.5, 0.5]])  # scale
        for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
            i *= 4
            if random.random() < 0.5:
                im1 = F.interpolate(im[i].unsqueeze(0).float(), scale_factor=2.0, mode='bilinear',
                                    align_corners=False)[0].type(im[i].type())

                im2 = F.interpolate(im2[i].unsqueeze(0).float(), scale_factor=2.0, mode='bilinear',
                                    align_corners=False)[0].type(im2[i].type())
                lb = label[i]
            else:
                im1 = torch.cat((torch.cat((im[i], im[i + 1]), 1), torch.cat((im[i + 2], im[i + 3]), 1)), 2)
                im2 = torch.cat((torch.cat((im2[i], im2[i + 1]), 1), torch.cat((im2[i + 2], im2[i + 3]), 1)), 2)
                lb = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            im4.append(im1)
            im42.append(im2)
            label4.append(lb)

        for i, lb in enumerate(label4):
            lb[:, 0] = i  # add target image index for build_targets()

        return torch.stack(im4, 0), torch.stack(im42, 0), torch.cat(label4, 0), path4, path42, shapes4, shapes42


# Ancillary functions --------------------------------------------------------------------------------------------------
def flatten_recursive(path=DATASETS_DIR / 'coco128'):
    # Flatten a recursive directory by bringing all files to top level
    new_path = Path(f'{str(path)}_flat')
    if os.path.exists(new_path):
        shutil.rmtree(new_path)  # delete output folder
    os.makedirs(new_path)  # make new output folder
    for file in tqdm(glob.glob(f'{str(Path(path))}/**/*.*', recursive=True)):
        shutil.copyfile(file, new_path / Path(file).name)


def extract_boxes(path=DATASETS_DIR / 'coco128'):  # from utils.dataloaders import *; extract_boxes()
    # Convert detection dataset into classification dataset, with one directory per class
    path = Path(path)  # images dir
    shutil.rmtree(path / 'classification') if (path / 'classification').is_dir() else None  # remove existing
    files = list(path.rglob('*.*'))
    n = len(files)  # number of files
    for im_file in tqdm(files, total=n):
        if im_file.suffix[1:] in IMG_FORMATS:
            # image
            im = cv2.imread(str(im_file))[..., ::-1]  # BGR to RGB
            h, w = im.shape[:2]

            # labels
            lb_file = Path(img2label_paths([str(im_file)])[0])
            if Path(lb_file).exists():
                with open(lb_file) as f:
                    lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels

                for j, x in enumerate(lb):
                    c = int(x[0])  # class
                    f = (path / 'classifier') / f'{c}' / f'{path.stem}_{im_file.stem}_{j}.jpg'  # new filename
                    if not f.parent.is_dir():
                        f.parent.mkdir(parents=True)

                    b = x[1:] * [w, h, w, h]  # box
                    # b[2:] = b[2:].max()  # rectangle to square
                    b[2:] = b[2:] * 1.2 + 3  # pad
                    b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(int)

                    b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
                    b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                    assert cv2.imwrite(str(f), im[b[1]:b[3], b[0]:b[2]]), f'box failure in {f}'


def autosplit(path=DATASETS_DIR / 'coco128/images', weights=(0.9, 0.1, 0.0), annotated_only=False):
    """ Autosplit a dataset into train/val/test splits and save path/autosplit_*.txt files
    Usage: from utils.dataloaders import *; autosplit()
    Arguments
        path:            Path to images directory
        weights:         Train, val, test weights (list, tuple)
        annotated_only:  Only use images with an annotated txt file
    """
    path = Path(path)  # images dir
    files = sorted(x for x in path.rglob('*.*') if x.suffix[1:].lower() in IMG_FORMATS)  # image files only
    n = len(files)  # number of files
    random.seed(0)  # for reproducibility
    indices = random.choices([0, 1, 2], weights=weights, k=n)  # assign each image to a split

    txt = ['autosplit_train.txt', 'autosplit_val.txt', 'autosplit_test.txt']  # 3 txt files
    for x in txt:
        if (path.parent / x).exists():
            (path.parent / x).unlink()  # remove existing

    print(f'Autosplitting images from {path}' + ', using *.txt labeled images only' * annotated_only)
    for i, img in tqdm(zip(indices, files), total=n):
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():  # check label
            with open(path.parent / txt[i], 'a') as f:
                f.write(f'./{img.relative_to(path.parent).as_posix()}' + '\n')  # add image to txt file


def verify_image_label(args):
    # Verify one image-label pair
    im_file, lb_file, prefix = args
    nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, '', []  # number (missing, found, empty, corrupt), message, segments
    try:
        # verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
        assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}'
        if im.format.lower() in ('jpg', 'jpeg'):
            with open(im_file, 'rb') as f:
                f.seek(-2, 2)
                if f.read() != b'\xff\xd9':  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, 'JPEG', subsampling=0, quality=100)
                    msg = f'{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved'

        # verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any(len(x) > 6 for x in lb):  # is segment
                    classes = np.array([x[0] for x in lb], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
                    lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                lb = np.array(lb, dtype=np.float32)
            nl = len(lb)
            if nl:
                assert lb.shape[1] == 5, f'labels require 5 columns, {lb.shape[1]} columns detected'
                assert (lb >= 0).all(), f'negative label values {lb[lb < 0]}'
                assert (lb[:, 1:] <= 1).all(), f'non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}'
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates
                    if segments:
                        segments = [segments[x] for x in i]
                    msg = f'{prefix}WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed'
            else:
                ne = 1  # label empty
                lb = np.zeros((0, 5), dtype=np.float32)
        else:
            nm = 1  # label missing
            lb = np.zeros((0, 5), dtype=np.float32)
        return im_file, lb, shape, segments, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f'{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}'
        return [None, None, None, None, nm, nf, ne, nc, msg]


class HUBDatasetStats():
    """ Class for generating HUB dataset JSON and `-hub` dataset directory

    Arguments
        path:           Path to data.yaml or data.zip (with data.yaml inside data.zip)
        autodownload:   Attempt to download dataset if not found locally

    Usage
        from utils.dataloaders import HUBDatasetStats
        stats = HUBDatasetStats('coco128.yaml', autodownload=True)  # usage 1
        stats = HUBDatasetStats('path/to/coco128.zip')  # usage 2
        stats.get_json(save=False)
        stats.process_images()
    """

    def __init__(self, path='coco128.yaml', autodownload=False):
        # Initialize class
        zipped, data_dir, yaml_path = self._unzip(Path(path))
        try:
            with open(check_yaml(yaml_path), errors='ignore') as f:
                data = yaml.safe_load(f)  # data dict
                if zipped:
                    data['path'] = data_dir
        except Exception as e:
            raise Exception('error/HUB/dataset_stats/yaml_load') from e

        check_dataset(data, autodownload)  # download dataset if missing
        self.hub_dir = Path(data['path'] + '-hub')
        self.im_dir = self.hub_dir / 'images'
        self.im_dir.mkdir(parents=True, exist_ok=True)  # makes /images
        self.stats = {'nc': data['nc'], 'names': list(data['names'].values())}  # statistics dictionary
        self.data = data

    @staticmethod
    def _find_yaml(dir):
        # Return data.yaml file
        files = list(dir.glob('*.yaml')) or list(dir.rglob('*.yaml'))  # try root level first and then recursive
        assert files, f'No *.yaml file found in {dir}'
        if len(files) > 1:
            files = [f for f in files if f.stem == dir.stem]  # prefer *.yaml files that match dir name
            assert files, f'Multiple *.yaml files found in {dir}, only 1 *.yaml file allowed'
        assert len(files) == 1, f'Multiple *.yaml files found: {files}, only 1 *.yaml file allowed in {dir}'
        return files[0]

    def _unzip(self, path):
        # Unzip data.zip
        if not str(path).endswith('.zip'):  # path is data.yaml
            return False, None, path
        assert Path(path).is_file(), f'Error unzipping {path}, file not found'
        unzip_file(path, path=path.parent)
        dir = path.with_suffix('')  # dataset directory == zip name
        assert dir.is_dir(), f'Error unzipping {path}, {dir} not found. path/to/abc.zip MUST unzip to path/to/abc/'
        return True, str(dir), self._find_yaml(dir)  # zipped, data_dir, yaml_path

    def _hub_ops(self, f, max_dim=1920):
        # HUB ops for 1 image 'f': resize and save at reduced quality in /dataset-hub for web/app viewing
        f_new = self.im_dir / Path(f).name  # dataset-hub image filename
        try:  # use PIL
            im = Image.open(f)
            r = max_dim / max(im.height, im.width)  # ratio
            if r < 1.0:  # image too large
                im = im.resize((int(im.width * r), int(im.height * r)))
            im.save(f_new, 'JPEG', quality=50, optimize=True)  # save
        except Exception as e:  # use OpenCV
            LOGGER.info(f'WARNING ⚠️ HUB ops PIL failure {f}: {e}')
            im = cv2.imread(f)
            im_height, im_width = im.shape[:2]
            r = max_dim / max(im_height, im_width)  # ratio
            if r < 1.0:  # image too large
                im = cv2.resize(im, (int(im_width * r), int(im_height * r)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(f_new), im)

    def get_json(self, save=False, verbose=False):
        # Return dataset JSON for Ultralytics HUB
        def _round(labels):
            # Update labels to integer class and 6 decimal place floats
            return [[int(c), *(round(x, 4) for x in points)] for c, *points in labels]

        for split in 'train', 'val', 'test':
            if self.data.get(split) is None:
                self.stats[split] = None  # i.e. no test set
                continue
            dataset = LoadImagesAndLabels(self.data[split])  # load dataset
            x = np.array([
                np.bincount(label[:, 0].astype(int), minlength=self.data['nc'])
                for label in tqdm(dataset.labels, total=dataset.n, desc='Statistics')])  # shape(128x80)
            self.stats[split] = {
                'instance_stats': {
                    'total': int(x.sum()),
                    'per_class': x.sum(0).tolist()},
                'image_stats': {
                    'total': dataset.n,
                    'unlabelled': int(np.all(x == 0, 1).sum()),
                    'per_class': (x > 0).sum(0).tolist()},
                'labels': [{
                    str(Path(k).name): _round(v.tolist())} for k, v in zip(dataset.im_files, dataset.labels)]}

        # Save, print and return
        if save:
            stats_path = self.hub_dir / 'stats.json'
            print(f'Saving {stats_path.resolve()}...')
            with open(stats_path, 'w') as f:
                json.dump(self.stats, f)  # save stats.json
        if verbose:
            print(json.dumps(self.stats, indent=2, sort_keys=False))
        return self.stats

    def process_images(self):
        # Compress images for Ultralytics HUB
        for split in 'train', 'val', 'test':
            if self.data.get(split) is None:
                continue
            dataset = LoadImagesAndLabels(self.data[split])  # load dataset
            desc = f'{split} images'
            for _ in tqdm(ThreadPool(NUM_THREADS).imap(self._hub_ops, dataset.im_files), total=dataset.n, desc=desc):
                pass
        print(f'Done. All images saved to {self.im_dir}')
        return self.im_dir


# Classification dataloaders -------------------------------------------------------------------------------------------
class ClassificationDataset(torchvision.datasets.ImageFolder):
    """
    YOLOv5 Classification Dataset.
    Arguments
        root:  Dataset path
        transform:  torchvision transforms, used by default
        album_transform: Albumentations transforms, used if installed
    """

    def __init__(self, root, augment, imgsz, cache=False):
        super().__init__(root=root)
        self.torch_transforms = classify_transforms(imgsz)
        self.album_transforms = classify_albumentations(augment, imgsz) if augment else None
        self.cache_ram = cache is True or cache == 'ram'
        self.cache_disk = cache == 'disk'
        self.samples = [list(x) + [Path(x[0]).with_suffix('.npy'), None] for x in self.samples]  # file, index, npy, im

    def __getitem__(self, i):
        f, j, fn, im = self.samples[i]  # filename, index, filename.with_suffix('.npy'), image
        if self.cache_ram and im is None:
            im = self.samples[i][3] = cv2.imread(f)
        elif self.cache_disk:
            if not fn.exists():  # load npy
                np.save(fn.as_posix(), cv2.imread(f))
            im = np.load(fn)
        else:  # read image
            im = cv2.imread(f)  # BGR
        if self.album_transforms:
            sample = self.album_transforms(image=cv2.cvtColor(im, cv2.COLOR_BGR2RGB))['image']
        else:
            sample = self.torch_transforms(im)
        return sample, j


def create_classification_dataloader(path,
                                     imgsz=224,
                                     batch_size=16,
                                     augment=True,
                                     cache=False,
                                     rank=-1,
                                     workers=8,
                                     shuffle=True):
    # Returns Dataloader object to be used with YOLOv5 Classifier
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = ClassificationDataset(root=path, imgsz=imgsz, augment=augment, cache=cache)
    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return InfiniteDataLoader(dataset,
                              batch_size=batch_size,
                              shuffle=shuffle and sampler is None,
                              num_workers=nw,
                              sampler=sampler,
                              pin_memory=PIN_MEMORY,
                              worker_init_fn=seed_worker,
                              generator=generator)  # or DataLoader(persistent_workers=True)
