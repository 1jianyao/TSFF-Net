from utils.torch_utils import select_device, smart_inference_mode
from models.common import DetectMultiBackend
# from utils.dataloaders4 import LoadImages3
import cv2
from sf6_predict import run
from torchstat import stat  # 查看网络参数
# Load model(导入模型)
device = ''
device = select_device(device)
# weights = r'./runs/train/exp2/weights/best.pt'
weights = r"D:\project1\sf6_dection\objection_detection4\yolov5-master_renew\mix_fusion94\exp94\weights\best.pt"
# weights =r"D:\project1\sf6_dection\objection_detection4\yolov5-master_renew\e\mix_fusion122\exp106\weights\best.pt"
dnn = False
data = "data/SF6_VOC.yaml"
half = False
model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
# print(model)
# Load model

# stride, names, pt = model.stride, model.names, model.pt
#
# imgsz = (640, 640),  # inference size (height, width)
# vid_stride=1,  # video frame-rate stride
# augment=False"D:\project1\sf6_dection\objection_detection4\yolov5-master_renew\yolov5-master\VOCData3\images2\c2-106.jpg"
# visualize=False
img1 = cv2.imread(r"./VOCData4\images\0-39.jpg")
# print("img_shape",img1.shape)
img2 = cv2.imread(r"./VOCData4\images2\0-39.jpg")
result=run(source=img1,source2=img2,model=model)
# cv2.imshow("fusion",result)
# cv2.imwrite("test.jpg",result)
# cv2.waitKey(0)
# dataset = LoadImages3(img1,img1, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
# for im, im0s,im2,im0s2 in dataset:
#     pred = model(im, im2, augment=augment, visualize=visualize)
#     pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
#     for i, det in enumerate(pred):  # per image
#         seen += 1
#         im0, frame = im0s.copy(), getattr(dataset, 'frame', 0)
#         annotator = Annotator(im0, line_width=line_thickness, example=str(names))
#         if len(det):
#             # Rescale boxes from img_size to im0 size
#             det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
#             # Print results
#             for c in det[:, 5].unique():
#                 n = (det[:, 5] == c).sum()  # detections per class
#
#             # Write results
#             for *xyxy, conf, cls in reversed(det):
#                 ###
#                 c = int(cls)  # integer class
#                 names = {0: "SF6"}
#                 label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
#                 annotator.box_label(xyxy, label, color=colors(c, True))
# import torchvision.transforms as transforms
# import cv2
# from torchvision import utils as vutils
#
# img1 = cv2.imread(
#     r'D:\project1\sf6_dection\SLBAF-Net-20.04\SLBAF-Net-20.04\modules\yolov5-dual\data\images\0-1.jpg')
# # print(img1.shape)  # numpy数组格式为（H,W,C）
#
# crop_size = (640, 640)
#
# img1 = cv2.resize(img1, crop_size, interpolation=cv2.INTER_CUBIC)
# transf = transforms.ToTensor()
# img1 = transf(img1)  # tensor数据格式是torch(C,H,W)
# # img1=img_tensor1.to(device, non_blocking=True).float() / 255
# img1 = img1.reshape(1, 3, 640, 640)
# # img1 = img1.to(device).float() / 255
#
# img2 = cv2.imread(
#     r'D:\project1\sf6_dection\SLBAF-Net-20.04\SLBAF-Net-20.04\modules\yolov5-dual\data\images2\0-5.jpg')
# crop_size = (640, 640)
#
# img2 = cv2.resize(img2, crop_size, interpolation=cv2.INTER_CUBIC)
#
# transf = transforms.ToTensor()
# img2 = transf(img2)  # tensor数据格式是torch(C,H,W)
# img2 = img2.reshape(1, 3, 640, 640)
# outputs = model(img1, img2)
# print(outputs.shape)