# import torch
# import os
# from models.yolo import *
# # os.chdir("/sf6_dection/SLBAF-Net-20.04/SLBAF-Net-20.04/modules/yolov5-dual/models")
# state_dict=torch.load(r"D:/project1/sf6_dection/SLBAF-Net-20.04/SLBAF-Net-20.04/modules/yolov5-dual/weights/yolov5s.pt")
# print(type(state_dict))
# # 既然是字典，查查字典内存储的是什么['epoch', 'best_fitness', 'model', 'ema', 'updates', 'optimizer', 'wandb_id', 'date']
# print(state_dict.keys())
# # # 只要模型权重
# model=state_dict["model"]
# print(type(model))
# print(model)


import torch
import os
from models.yolo import *
# os.chdir("/sf6_dection/SLBAF-Net-20.04/SLBAF-Net-20.04/modules/yolov5-dual/models")


import argparse
import torch
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=r"D:\project1\sf6_dection\objection_detection4\yolov5-master - 复现\runs\train\weights\last.pt", help='输入的权重.pt文件')
    parser.add_argument('--ToFP16', action='store_true', help='convert model from FP32 to FP16')
    opt = parser.parse_args()

    model = torch.load(opt.weights, map_location=torch.device('cpu'))
    # 读取模型网络
    net = model['model']
    print(net)
# state_dict=torch.load(r"D:\project1\sf6_dection\objection_detection4\yolov5-master_renew\yolov5-master\runs\train\exp10\weights\last.pt")
# print(type(state_dict))
# # 既然是字典，查查字典内存储的是什么['epoch', 'best_fitness', 'model', 'ema', 'updates', 'optimizer', 'wandb_id', 'date']
# print(state_dict.keys())
# # # 只要模型权重
# model=state_dict["model"]
# print(type(model))
# print(model)
# #
# model=model.float()
# x=torch.zeros((1,3,640,640),dtype=torch.float32)
# #
# y=model(x)
# print("y:",type(y),len(y))
# print("这里的y有两个输出，分别是训练模式和推理模式，推理模式的输出是y[0],训练模式的输出是y[1]")
# print("y[0]",type(y[0]),y[0].shape,"\ny[1]:",type(y[1]),len(y[1]),"\n",y[1][0].shape,y[1][1].shape)
