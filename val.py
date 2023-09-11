# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Validate a trained YOLOv5 detection model on a detection dataset

Usage:
    $ python val.py --weights yolov5s.pt --data coco128.yaml --img 640

Usage - formats:
    $ python val.py --weights yolov5s.pt                 # PyTorch
                              yolov5s.torchscript        # TorchScript
                              yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                              yolov5s_openvino_model     # OpenVINO
                              yolov5s.engine             # TensorRT
                              yolov5s.mlmodel            # CoreML (macOS-only)
                              yolov5s_saved_model        # TensorFlow SavedModel
                              yolov5s.pb                 # TensorFlow GraphDef
                              yolov5s.tflite             # TensorFlow Lite
                              yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                              yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.general import (LOGGER, TQDM_BAR_FORMAT, Profile, check_dataset, check_img_size, check_requirements,
                           check_yaml, coco80_to_coco91_class, colorstr, increment_path, non_max_suppression,
                           print_args, scale_boxes, xywh2xyxy, xyxy2xywh)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, smart_inference_mode


def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({
            'image_id': image_id,
            'category_id': class_map[int(p[5])],
            'bbox': [round(x, 3) for x in b],
            'score': round(p[4], 5)})


def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


#run 函数其实用train.py执行的，并不是执行val.py。
    """
    :params data: 数据集配置文件地址 包含数据集的路径、类别个数、类名、下载地址等信息 train.py时传入data_dict
    :params weights: 模型的权重文件地址 运行train.py=None 运行test.py=默认weights/yolov5s.pt
    :params batch_size: 前向传播的批次大小 运行test.py传入默认32 运行train.py则传入batch_size // WORLD_SIZE * 2
    :params imgsz: 输入网络的图片分辨率 运行test.py传入默认640 运行train.py则传入imgsz_test
    :params conf_thres: object置信度阈值 默认0.25
    :params iou_thres: 进行NMS时IOU的阈值 默认0.6
    :params task: 设置测试的类型 有train, val, test, speed or study几种 默认val
    :params device: 测试的设备
    :params single_cls: 数据集是否只用一个类别 运行test.py传入默认False 运行train.py则传入single_cls
    :params augment: 测试是否使用TTA Test Time Augment 默认False
    :params verbose: 是否打印出每个类别的mAP 运行test.py传入默认Fasle 运行train.py则传入nc < 50 and final_epoch
    :params save_txt: 是否以txt文件的形式保存模型预测框的坐标 默认True
    :params save_hybrid: 是否save label+prediction hybrid results to *.txt  默认False
    :params save_conf: 是否保存预测每个目标的置信度到预测tx文件中 默认True
    :params save_json: 是否按照coco的json格式保存预测框，并且使用cocoapi做评估（需要同样coco的json格式的标签）
                       运行test.py传入默认Fasle 运行train.py则传入is_coco and final_epoch(一般也是False)
    :params project: 测试保存的源文件 默认runs/test
    :params name: 测试保存的文件地址 默认exp  保存在runs/test/exp下
    :params exist_ok: 是否存在当前文件 默认False 一般是 no exist-ok 连用  所以一般都要重新创建文件夹
    :params half: 是否使用半精度推理 FP16 half-precision inference 默认False
    :params model: 模型 如果执行test.py就为None 如果执行train.py就会传入ema.ema(ema模型)
    :params dataloader: 数据加载器 如果执行test.py就为None 如果执行train.py就会传入testloader
    :params save_dir: 文件保存路径 如果执行test.py就为‘’ 如果执行train.py就会传入save_dir(runs/train/expn)
    :params plots: 是否可视化 运行test.py传入默认True 运行train.py则传入plots and final_epoch
    :params wandb_logger: 网页可视化 类似于tensorboard 运行test.py传入默认None 运行train.py则传入wandb_logger(train)
    :params compute_loss: 损失函数 运行test.py传入默认None 运行train.py则传入compute_loss(train)
    :return (Precision, Recall, map@0.5, map@0.5:0.95, box_loss, obj_loss, cls_loss)
    """
@smart_inference_mode()
def run(
        data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        max_det=300,  # maximum detections per image
        task='val',  # train, val, test, speed or study
        task2='val2',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        workers=8,  # max dataloader workers (per RANK in DDP mode)
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
):
    # ——————————————————————————————————————————————————————————————————————
    #                       一、初始化配置
    # ——————————————————————————————————————————————————————————————————————
    # 1.1 初始化模型参数
    # ——————————————————————————————————————————————————————————————————————
    # Initialize/load model and set device
    # 训练时（train.py）调用：初始化模型参数，训练设备
    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()
    #验证时（val.py）调用：初始化设备、save_dir文件路径、make dir、加载模型、check imgsz、 加载+check data配置信息
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        # 生成save_dir文件路径 run/test/exp
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        # 加载模型load FP32 model 只在运行test.py才需要自己加载model
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        # 检测输入图像分辨率imgsz是否能被gs整除，只在运行test.py才需要自己生成check imgsz
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        half = model.fp16  # FP16 supported on limited backends with CUDA
        if engine:
            batch_size = model.batch_size
        else:
            device = model.device
            if not (pt or jit):
                batch_size = 1  # export.py models default to batch-size 1
                LOGGER.info(f'Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models')

        # Data
        data = check_dataset(data)  # check
# 1.2初始化map
# ——————————————————————————————————————————————————————————————————————
    # Configure
    model.eval()
    cuda = device.type != 'cpu'
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith(f'coco{os.sep}val2017.txt')  # COCO dataset
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
# ——————————————————————————————————————————————————————————————————————
#                       二、加载数据集
# ——————————————————————————————————————————————————————————————————————
# Dataloader
# 如果不是训练(执行val.py脚本调用run函数)就调用create_dataloader生成dataloader
# 如果是训练(执行train.py调用run函数)就不需要生成dataloader 可以直接从参数中传过来valloader
    if not training:
        # if pt and not single_cls:  # check --weights are trained on --data
        #     ncm = model.model.nc
        #     assert ncm == nc, f'{weights} ({ncm} classes) trained on different --data than what you passed ({nc} ' \
        #                       f'classes). Pass correct combination of --weights and --data that are trained together.'
        model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
        pad, rect = (0.0, False) if task == 'speed' else (0.5, pt)  # square inference for benchmarks
        task = task if task in ('train', 'val', 'test','test11','test12','test13','test14') else 'val'  # path to train/val/test images
        # 这边需要修改
        task2 = task2 if task2 in ('train2', 'val2', 'test2','test21','test22','test23','test24') else 'val2'  # path to train/val/test images
        ################ 感觉这里有问题         ######################################
        dataloader = create_dataloader(data[task],
                                       data[task2],
                                       imgsz,
                                       batch_size,
                                       stride,
                                       single_cls,
                                       pad=pad,
                                       rect=rect,
                                       workers=workers,
                                       prefix=colorstr(f'{task}: '))[0]
#   2.1 初始化测试需要的参数
    seen = 0  # 初始化测试的图片数量
    confusion_matrix = ConfusionMatrix(nc=nc) # 初始化混淆矩阵
    # 获取数据集所有类别的类名
    names = model.names if hasattr(model, 'names') else model.module.names  # get class names

    if isinstance(names, (list, tuple)):  # old format
        names = dict(enumerate(names))
    # 获取coco数据集的类别索引
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    # 设置tqdm进度条的显示信息
    s = ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
    # 初始化tp, fp,r,f1,mp,mr,map50,map指标和时间t0,t1,t2
    tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    dt = Profile(), Profile(), Profile()  # profiling times
    loss = torch.zeros(3, device=device)
    # 初始化json文件在的字典，统计信息 ap等
    jdict, stats, ap, ap_class = [], [], [], []
    callbacks.run('on_val_start')

# ——————————————————————————————————————————————————————————————————————
#                       三、开始验证
# ——————————————————————————————————————————————————————————————————————
# 3.1 图片预处理
# ——————————————————————————————————————————————————————————————————————
    #  bar_format=TQDM_BAR_FORMAT这个参数可能也得修改
    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)  # progress bar

######################################################################
    for batch_i, (im,im2,targets, paths, path2,shapes,shape2) in enumerate(pbar):
        callbacks.run('on_val_batch_start')
        with dt[0]:
            if cuda:
                im = im.to(device, non_blocking=True)
                im2 = im2.to(device, non_blocking=True)
                targets = targets.to(device)
            # 如果half为True,就把图片变为half精度
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0

            im2 = im2.half() if half else im2.float()  # uint8 to fp16/32
            im2 /= 255  # 0 - 255 to 0.0 - 1.0
            nb, _, height, width = im.shape  # batch size, channels, height, width
            nb2, _, height2, width2 = im2.shape  # batch size, channels, height, width
        # Inference
        with dt[1]:
            # 传入图像并计算目标框结果（这里也注意一下）
            preds, train_out = model(im,im2) if compute_loss else (model(im,im2, augment=augment), None)
            # preds, train_out = model(im2, im) if compute_loss else (model(im2, im, augment=augment), None)
        # Loss
        if compute_loss:
            # 预测目标框结果和实际目标框结果进行对比，计算三个损失函数
            loss += compute_loss(train_out, targets)[1]  # box, obj, cls

        # NMS
        # ——————————————————————————————————————————————————————————————————————
        # NMS非极大值抑制,在检测算法中用来去掉大量有重叠的检测框,保留下来质量最高的框
        # 将真实框target的xywh(因为target是在labelimg中做了归一化的)映射到img(test)尺寸
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        with dt[2]:
            preds = non_max_suppression(preds,
                                        conf_thres,
                                        iou_thres,
                                        labels=lb,
                                        multi_label=True,
                                        agnostic=single_cls,
                                        max_det=max_det)
        # 3.5 统计真实框、预测框信息
        # ——————————————————————————————————————————————————————————————————————
        # Metrics
        # 为每张图片做统计，写入预测信息到txt文件，生成json文件字典，统计tp等
        # out: list{bs}  [300, 6] [42, 6] [300, 6] [300, 6]  [:, image_index+class+xywh]
        # Metrics
        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1
            # 如果预测为空，则添加空的信息到stats里
            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                    if plots:
                        confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            # 将预测坐标映射导原图img中
            scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

            # 保存预测信息导image_name.txt文件
            # Save/log
            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / f'{path.stem}.txt')
            if save_json:
                save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
            callbacks.run('on_val_image_end', pred, predn, path, names, im[si])

        # Plot images
        if plots and batch_i < 3:
            plot_images(im, targets, paths, save_dir / f'val_batch{batch_i}_labels.jpg', names)  # labels
            plot_images(im, output_to_target(preds), paths, save_dir / f'val_batch{batch_i}_pred.jpg', names)  # pred

        callbacks.run('on_val_batch_end', batch_i, im, targets, paths, shapes, preds)
# ——————————————————————————————————————————————————————————————————————
#                       四、计算map
# ——————————————————————————————————————————————————————————————————————
# 4.1 计算各项指标
# ——————————————————————————————————————————————————————————————————————
    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class
# 4.2 打印结果
# ——————————————————————————————————————————————————————————————————————
    # Print results
    pf = '%22s' + '%11i' * 2 + '%11.3g' * 4  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    if nt.sum() == 0:
        LOGGER.warning(f'WARNING ⚠️ no labels found in {task} set, can not compute metrics without labels')

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run('on_val_end', nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix)

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = str(Path('instances_val2017.json'))  # annotations
        pred_json = str(save_dir / f'{w}_predictions.json')  # predictions
        LOGGER.info(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            check_requirements('pycocotools>=2.0.6')
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.im_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            LOGGER.info(f'pycocotools unable to run: {e}')
# ——————————————————————————————————————————————————————————————————————
#                       五、返回测试结果
# ——————————————————————————————————————————————————————————————————————
    # Return results

    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/SF6_VOC.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / r'D:\project1\sf6_dection\objection_detection4\yolov5-master_renew2\mix_fusionsp_conv\exp161\weights\best.pt', help='model path(s)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=300, help='maximum detections per image')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--task2', default='val2', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true',default=True, help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))

    if opt.task in ('train', 'val', 'test','test11','test12','test13','test14'):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(f'WARNING ⚠️ confidence threshold {opt.conf_thres} > 0.001 produces invalid results')
        if opt.save_hybrid:
            LOGGER.info('WARNING ⚠️ --save-hybrid will return high mAP from hybrid labels, not from predictions alone')
        run(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = torch.cuda.is_available() and opt.device != 'cpu'  # FP16 for fastest results
        if opt.task == 'speed':  # speed benchmarks
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        elif opt.task == 'study':  # speed vs mAP benchmarks
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            for opt.weights in weights:
                f = f'study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt'  # filename to save to
                x, y = list(range(256, 1536 + 128, 128)), []  # x axis (image sizes), y axis
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f'\nRunning {f} --imgsz {opt.imgsz}...')
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # results and times
                np.savetxt(f, y, fmt='%10.4g')  # save
            subprocess.run(['zip', '-r', 'study.zip', 'study_*.txt'])
            plot_val_study(x=x)  # plot
        else:
            raise NotImplementedError(f'--task {opt.task} not in ("train", "val", "test", "speed", "study")')


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
