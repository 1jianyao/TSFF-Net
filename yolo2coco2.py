import os
import cv2
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', default=r'D:\project1\sf6_dection\objection_detection4\yolov5-master_renew2\yolov5-master\VOCData', type=str,
                    help="root path of images and labels, include ./images and ./labels and classes.txt")
parser.add_argument('--save_path', type=str, default='instances_val2017.json',
                    help="if not split the dataset, give a path to a json file")

arg = parser.parse_args()

def get_str_btw(s, f, b):
    '''提取字符串s中，字符串f和b的中间部分'''
    par = s.partition(f)
    return (par[2].partition(b))[0][:]

def yolo2coco(arg):
    root_path = arg.root_dir
    print("Loading data from ", root_path)
    print()
    assert os.path.exists(root_path)
    # originLabelsDir = os.path.join(root_path, 'val_txt')
    #
    # originImagesDir = os.path.join(root_path, 'val')
    # 获取需要模型测试的标签和图像的地址列表

    val_path=os.path.join(root_path,"dataSet_path")
    # 获取地址
    val_path_txt=val_path+"/val.txt"
    f = open(val_path_txt, "r", encoding='UTF-8')
    # 图片地址
    content_image=f.readlines()
    label_path=os.path.join(root_path,"labels")
    # 标签列表
    content_label=[]
    # 图像名列表
    indexes=[]
    for con in content_image:
        # print(con)
        label_con=get_str_btw(con,"images/",".jpg")+".txt"
        label_con=os.path.join(label_path,label_con)
        images=get_str_btw(con,"images/",".jpg")+".jpg"
        content_label.append(label_con)
        indexes.append(images)
        # print(label_con)

    with open(os.path.join(root_path, 'classes.txt')) as f:
        classes = list(map(lambda x: x.strip(), f.readlines()))
    # images dir name
    # indexes = os.listdir(originImagesDir)
    # print(indexes)
    # exit()
    dataset = {'categories': [], 'annotations': [], 'images': []}
    for i, cls in enumerate(classes, 0):
        dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})

    # 标注的id
    ann_id_cnt = 0
    for k, index in enumerate(tqdm(indexes)):
        # 支持 png jpg 格式的图片。
        txtFile = index.replace('images', 'txt').replace('.jpg', '.txt').replace('.png', '.txt')
        # # 读取图像的宽和高
        # img_path=os.path.join(originImagesDir, index)
        # print(k)
        # print("img_path",img_path)
        # exit()
        # print("label改变前", os.path.join(originImagesDir, index),type(os.path.join(originImagesDir, index)))
        # print("label改变后", content_image[k].strip("\n"),type( content_image[k].strip("\n")))
        im = cv2.imread(content_image[k].strip("\n"))
        height, width, _ = im.shape
        label_path = content_label[k]

        # print("label改变前", os.path.join(originLabelsDir, txtFile))
        #
        # print("label改变后", label_path)
        # 添加图像的信息D:\project1\sf6_dection\objection_detection3\Fusion5_5_yolov5\VOCData5_5\labels\0-10.txt
        # if not os.path.exists(os.path.join(originLabelsDir, txtFile)):
        #     # 如没标签，跳过，只保留图片信息。
        #     continue
        dataset['images'].append({'file_name': index,
                                  'id': int(index[:-4]) if index[:-4].isnumeric() else index[:-4],
                                  'width': width,
                                  'height': height})
        with open(label_path, 'r',encoding="utf-8") as fr:
            labelList = fr.readlines()
            # print("label",labelList)
            # exit()
            for label in labelList:
                label = label.strip().split()
                x = float(label[1])
                y = float(label[2])
                w = float(label[3])
                h = float(label[4])

                # convert x,y,w,h to x1,y1,x2,y2
                H, W, _ = im.shape
                x1 = (x - w / 2) * W
                y1 = (y - h / 2) * H
                x2 = (x + w / 2) * W
                y2 = (y + h / 2) * H
                # 标签序号从0开始计算, coco2017数据集标号混乱，不管它了。
                cls_id = int(label[0])
                width = max(0, x2 - x1)
                height = max(0, y2 - y1)
                dataset['annotations'].append({
                    'area': width * height,
                    'bbox': [x1, y1, width, height],
                    'category_id': cls_id,
                    'id': ann_id_cnt,
                    'image_id': int(index[:-4]) if index[:-4].isnumeric() else index[:-4],
                    'iscrowd': 0,
                    # mask, 矩形是从左上角点按顺时针的四个顶点
                    'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
                })
                ann_id_cnt += 1

    # 保存结果
    with open(arg.save_path, 'w') as f:
        json.dump(dataset, f)
        print('Save annotation to {}'.format(arg.save_path))


if __name__ == "__main__":
    yolo2coco(arg)