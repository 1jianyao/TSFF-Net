# 用yolov5 完成目标检测任务
import time

import numpy as np
import cv2

from PIL import Image
import sys
sys.path.append(r'D:\project1\sf6_dection\classification\小论文/')
from color_transform import cut
from color_transform import HSV_analyze
from xml_and_iou import get_bndboxfromxml,calculate_iou,iou_img

from utils.torch_utils import select_device, smart_inference_mode
from models.common import DetectMultiBackend
import cv2
from sf6_predict2 import run
# # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def detection_sf6(video):
    global number
    global i
    global MP_num
    global FP_num
    global t_num
    global state
    global num_number
    global gt_box
    global inference_time

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    camera = cv2.VideoCapture(video)
    history = 20
    bs = cv2.createBackgroundSubtractorKNN(detectShadows=True)
    bs.setHistory(history)
    h = 0



    while True:
        # res为Ture或Flase，代表有没有读到帧图片(有没有读到图片信息)，frame1代表该帧图片
        res, frames1 = camera.read()
        if not res:
            break

        frames2 = cv2.cvtColor(frames1, cv2.COLOR_BGR2GRAY) # 数据格式为np.ndarray
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
        frames2 = clahe.apply(frames2)
        ## 需要使用的轮廓信息，和原始图片信息

        # RGB图转成伪RGB图像
        grad2__ = Image.fromarray(np.uint8(frames2))   # np.ndarray转化成PIL image
        # hash0 = imagehash.average_hash(grad2__)
        frame2 = grad2__.convert("RGB") # 灰度图转化成伪RGB图像
        # PIL image重新转化成 np.ndarray
        frames = np.array(frame2)
        # 双边滤波
        frames = cv2.bilateralFilter(frames, 9, 75, 75)
        # cv2.imshow("frames",frames)
        mask = np.zeros([frames.shape[0], frames.shape[1]], dtype=np.uint8)
        mask[70:479, :] = 255


        # 得到掩盖后的图像
        frame2 = cv2.add(frames, np.zeros(np.shape(frames), dtype=np.uint8), mask=mask)
        cimg = np.zeros_like(frames) + 255
        # 全黑图像
        cimg2 = np.zeros_like(frames)
        time_sta=time.time()
        fg_mask = bs.apply(frame2)

        if h < history:
            h += 1
            continue

        img_close = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        # # # # img=cv2.cvtColor(img,)
        th = cv2.threshold(img_close.copy(), 25, 255, cv2.THRESH_BINARY)[1]
        # 膨胀处理
        dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 4)), iterations=3)
        # 得到轮廓信息
        contour, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # 得到前景图像,使用预处理后的图像
        hsv1 = HSV_analyze(fg_mask, frames)
        # cv2.imshow("hsv1",hsv1)
        # # 全白图像

        for c in contour:

            # cv2.imshow("cimg", cimg)
            cv2.drawContours(frames1, [c], -1, (0, 0, 255), 2)
            cut_img = cut(cimg,c)
            # 取并集
            cimg2 = cv2.bitwise_or(cimg2, cut_img)

        # 取交集
        pixl_fore = cv2.bitwise_and(hsv1, cimg2)
        time_end = time.time()
        inference_time1=time_end-time_sta
        print("提取前景像素时间s",inference_time1)
        # 图像融和
        # fusion_img=Image_fusion(frames,pixl_fore,a,b)
        result,box_all,inference_time2=run(source=frames,source2=pixl_fore,model=model)
        if  i>0:
            inference_time.append(inference_time1+inference_time2)
        i += 1
        print("i",i)
        # print("box_all", box_all[])
        # print(out_file4 + "/" + "{}-".format(video_n)+"{}.jpg".format(i))
        # cv2.imwrite(out_file4 + "/" + "c{}-".format(video_n)+"{}.jpg".format(i), result)
        # cv2.imwrite(out_file4_fpi + "/" + "c{}-".format(video_n)+"{}.jpg".format(i), pixl_fore)



        # 真实边框
        if "sf6_bin" in video:
            xml_path = ann_path + "/" + "{}-".format(video_n) + "{}.xml".format(i)
        else:
            xml_path= ann_path + "/" + "c{}-".format(video_n) + "{}.xml".format(i)
        # print("xml_path",xml_path)
        if os.path.exists(xml_path):
            # 说明有标签文件
            gt_box += 1
        else:
            continue
        bonbox = get_bndboxfromxml(xml_path)
        # 误检率和漏检率

        # 不存在IOU>0.5的，漏检
        MP=True
        # 存在IOU<0.5的，误检
        FP=False
        IOU_max_MP=0
        frames2, FP, IOU_max_MP=iou_img(bonbox,box_all,frames,IOU_max_MP,FP,thresh)
        # print(out_file4 + "/" + "{}-".format(video_n)+"{}.jpg".format(i))
        # cv2.imwrite(out_file4 + "/" + "c{}-".format(video_n)+"{}.jpg".format(i), result)

        cv2.imshow("result", result)
        # cv2.imshow("pixl_fore", pixl_fore)
        # cv2.imshow("frames", frames)
        print("IOU_max",IOU_max_MP)
        IOU_max_MP_all.append(IOU_max_MP)
        if  IOU_max_MP>thresh:
            MP=False
        if  MP:
            Ma_pic.append("{}-".format(video_n)+"{}.jpg".format(i))
            MP_num+=1
        if  FP:
            FP_num+=1

        # cv2.namedWindow("frames1", 0)
        # cv2.resizeWindow("frames1", 576, 400)
        # cv2.imshow("frames1",frames1)

        # 设置停留时间和停止按键
        if cv2.waitKey(6) & 0xff == ord("q"):
            break
    # 停止视频播放
    camera.release()




if __name__ == '__main__':
    import os
    # Load model(导入模型)
    device = ''
    device = select_device(device)
    weights= r'D:\project1\sf6_dection\objection_detection4\yolov5-master_renew2\mix_fusionsp_conv\exp161\weights\best.pt'
    dnn = False
    data = "data/SF6_VOC.yaml"
    half = False
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)

    IOU_max_MP_all=[]
    # 阈值
    thresh=0.2
    # thresh=0.1
    thresh2=0.5
    # 视频集
    # get_box
    gt_box=0

    MP_num=0
    FP_num=0


    t_num=0
    # 总样本
    iou_all = []

    Ma_pic=[]

    # 推理时间
    inference_time=[]
    # detection_sf6(video_path, model1)
    ann_path = r"D:\project1\sf6_dection\objection_detection3\Fusion5_5_yolov5\VOCData5_5\Annotations"

    video_numbers=18
    for video_n in range(video_numbers-1,video_numbers):
        video_path = r"D:\project1\sf6_dection\classification\data\sf6_bin" + "/" + str(video_n) + ".AVI"
        # video_path = r"D:\project1\sf6_dection\classification\data\SF6_Leak\sf6_color" + "/" + str(video_n) + ".AVI"
    #
        # out_file4 = r"D:\project1\sf6_dection\objection_detection4\yolov5-master_renew\yolov5-master\runs/FA-MA"+ "/" + str(video_n)
        # # out_file4_fpi = r"D:\project1\sf6_dection\objection_detection4\yolov5-master_renew\yolov5-master\runs/test_fpi" + "/" + str(
        # #     video_n)
        # if not os.path.exists(out_file4):
        #     os.makedirs(out_file4)
        # if not os.path.exists(out_file4_fpi):
        #     os.makedirs(out_file4_fpi)
        # 定义图像个数
        i = 0
        detection_sf6(video_path)
    print("标签文件的个数",gt_box)
    print("漏检个数个数", MP_num)
    print("误检个数", FP_num)

    print("误报率", FP_num/gt_box)
    print("漏报率", MP_num/gt_box)
    print("平均推理时间",np.mean(inference_time))
    print("FPS:{:2f}".format(1/np.mean(inference_time)))

    print("漏检图像名",Ma_pic)
    # i=0
    print("iou_all",IOU_max_MP_all)
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('TkAgg')
    # # x=range(0, len(iou_all))
    #
    x=list(range(0, len(IOU_max_MP_all)))
    #
    # # FP_num5=[x if x<0.5 else  for x in iou_all]
    print("总检测个数", len(IOU_max_MP_all))
    FP_num5 = [i for i in IOU_max_MP_all if i < thresh]
    print("漏检个数",len(FP_num5))
    # print("数据长度",len(iou_all))
    #
    # # plt.plot(x,iou_all)
    plt.scatter(x, IOU_max_MP_all, s=50)
    # plt.show()
    y=thresh*np.ones([len(IOU_max_MP_all),1])
    # b=[1,2,3]
    plt.plot(x,y)
    plt.show()
    # print(1)
