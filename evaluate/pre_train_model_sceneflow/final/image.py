import numpy as np
from PIL import Image
from os.path import *
import re
import json
import imageio
import cv2
import sys

import os


def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(rb'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(elapse):
    elapse = int(elapse)
    hour = elapse // 3600
    minute = elapse % 3600 // 60
    seconds = elapse % 60
    return "{:02d}:{:02d}:{:02d}".format(hour, minute, seconds)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def readDispCRES(filename):
    disp = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    disp = disp.astype(np.float32) / 32
    valid = np.abs(disp) > 0.0
    return disp, valid


# # 提取crestereo disp数据集
# image_path = "E:\\yxz\\dataset\\sample_cres_result\\shapenet0"
#
# files=os.listdir(image_path)
# for file in files:
#     path = image_path+"\\"+file
#     # disp0 = cv2.imread(path)
#     a=file[-8:]
#     if a=="disp.png":
#         disp,v=readDispCRES(path)
#
#         save_path0="E:/yxz/shiyan/shapenet0/"
#         ensure_dir(save_path0)
#         save_path=save_path0+file
#         # cv2.imshow("disp",disp)
#         cv2.imwrite(save_path,disp)

image_path = ""

files = os.listdir(image_path)
for file in files:
    path = image_path + "\\" + file
    # disp0 = cv2.imread(path)
    a = file[-8:]
    if a == "disp.png":
        disp, v = readDispCRES(path)

        save_path0 = ""
        ensure_dir(save_path0)
        save_path = save_path0 + file
        # cv2.imshow("disp",disp)
        cv2.imwrite(save_path, disp)
