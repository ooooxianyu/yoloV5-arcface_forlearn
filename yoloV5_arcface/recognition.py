from __future__ import print_function
import os
import cv2
from arc_face import *
import torch
import numpy as np
import time
#from config import Config
from torch.nn import DataParallel

def cv2_letterbox_image(image, expected_size):
    ih, iw = image.shape[0:2]
    ew, eh = expected_size,expected_size
    scale = min(eh / ih, ew / iw) # 最大边缩放至416得比例
    nh = int(ih * scale)
    nw = int(iw * scale)
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC) # 等比例缩放，使得有一边416
    top = (eh - nh) // 2 # 上部分填充的高度
    bottom = eh - nh - top  # 下部分填充的高度
    left = (ew - nw) // 2 # 左部分填充的距离
    right = ew - nw - left # 右部分填充的距离
    # 边界填充
    new_img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)
    return new_img

def cosin_metric(x1, x2):
    #计算余弦距离
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def load_image(img_path):
    image = cv2.imread(img_path, 0)
    if image is None:
        return None
    image = cv2_letterbox_image(image,128)
    image = np.dstack((image, np.fliplr(image)))
    image = image.transpose((2, 0, 1))
    image = image[:, np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5
    return image


if __name__ == '__main__':

    #opt = Config.MYconfig()

    model = resnet_face18(False)

    model = DataParallel(model)
    # load_model(model, opt.test_model_path)
    model.load_state_dict(torch.load('weights/resnet18_110.pth'), strict=False)
    model.to(torch.device("cuda"))

    image = load_image("inference/ouyanana.jpg")
    print(image.shape)

    data = torch.from_numpy(image)
    data = data.to(torch.device("cuda"))
    output = model(data)  # 获取特征
    output = output.data.cpu().numpy()
    print(output.shape)

    # 获取不重复图片 并分组
    fe_1 = output[::2]
    fe_2 = output[1::2]
    # print("this",cnt)
    # print(fe_1.shape,fe_2.shape)
    feature = np.hstack((fe_1, fe_2))
    feature = feature.reshape(1024)
    print(feature.shape)

    person_dict = {}
    person_dict["ouyannana"] = feature
    print(person_dict)

