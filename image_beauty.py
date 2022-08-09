import sys
#sys.path.insert(0, "d:\\apps\\nlp\\prompt\\modnet_docker")

import cv2
import paddlehub as hub
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
import os
from PIL import Image
import glob
from copy import deepcopy
import logging
from numba import jit

enlarge_eyes_radius = 15         # 眼睛放大范围
# enlarge_eyes_strength = 10       # 眼睛放大程度


# PaddleHub 获取特征点
def get_hub_module():
    return hub.Module(name="face_landmark_localization")


# 双线性插值法
@jit
def bilinear_insert(image, new_x, new_y):
    w, h, c = image.shape
    if c == 3:
        x1 = int(new_x)
        x2 = x1 + 1
        y1 = int(new_y)
        y2 = y1 + 1

        part1 = image[y1, x1].astype(np.float) * (float(x2) - new_x) * (float(y2) - new_y)
        part2 = image[y1, x2].astype(np.float) * (new_x - float(x1)) * (float(y2) - new_y)
        part3 = image[y2, x1].astype(np.float) * (float(x2) - new_x) * (new_y - float(y1))
        part4 = image[y2, x2].astype(np.float) * (new_x - float(x1)) * (new_y - float(y1))

        insert_value = part1 + part2 + part3 + part4

        return insert_value.astype(np.int8)

@jit
def lift_face_alg(image, start_point, end_point, radius, degree):
    #logging.info("min_dist:" + str(min_dist))
    radius_square = math.pow(radius, 2)
    image_cp = image.copy()

    dist_se = math.pow(np.linalg.norm(end_point - start_point), 2)
    height, width, channel = image.shape

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, tuple(start_point), np.int32(radius), 255, cv2.FILLED)

    xy = np.where(mask > 0)
    for j, i in zip(*xy):
        distance = (i - start_point[0]) * (i - start_point[0]) + (j - start_point[1]) * (j - start_point[1])
        #if distance < radius_square:
            # 计算出（i,j）坐标的原坐标
            # 计算公式中右边平方号里的部分
        ratio = (radius_square - distance) / (radius_square - distance + dist_se)
        ratio = ratio * ratio * degree * 0.8 / 100

        # if min_dist <= 120:
        #     ratio = ratio * min_dist / 120

        # 映射原位置
        new_x = i - ratio * (end_point[0] - start_point[0])
        new_y = j - ratio * (end_point[1] - start_point[1])

        new_x = new_x if new_x >= 0 else 0
        new_x = new_x if new_x < height - 1 else height - 2
        new_y = new_y if new_y >= 0 else 0
        new_y = new_y if new_y < width - 1 else width - 2

        # 根据双线性插值法得到new_x, new_y的值
        image_cp[j, i] = bilinear_insert(image, new_x, new_y)

    return image_cp


# 瘦脸
@jit
def thin_face(image, face_landmark, ratio=1):
    end_point = face_landmark[30]

    # 瘦左脸，3号点到5号点的距离作为瘦脸距离
    dist_left = np.linalg.norm(face_landmark[3] - face_landmark[5])
    # 瘦右脸，13号点到15号点的距离作为瘦脸距离
    dist_right = np.linalg.norm(face_landmark[13] - face_landmark[15])

    min_dist = dist_left if dist_left < dist_right else dist_right

    image = lift_face_alg(image, face_landmark[3], end_point, dist_left, ratio)
    image = lift_face_alg(image, face_landmark[13], end_point, dist_right, ratio)
    return image


@jit
def get_new_pt(pt_X, pt_C, R, scaleRatio):
    dis_C_X = np.sqrt(np.dot((pt_X-pt_C),(pt_X-pt_C)))

    alpha = 1.0 - scaleRatio * pow(dis_C_X / R - 1.0, 2.0)

    return pt_C + alpha*(pt_X-pt_C)


@jit
def localScaleWap(img, pt_C, R, scaleRatio):
    h, w, c = img.shape
    # 文件拷贝
    copy_img = img.copy()

    # 创建蒙板
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, tuple(pt_C), np.int32(R), 255, cv2.FILLED)

    pt_C = np.float32(pt_C)
    xy = np.where(mask > 0)
    for j, i in zip(*xy):
        pt_X = np.array([i, j], dtype=np.float32)

        pt_U = get_new_pt(pt_X, pt_C, R, scaleRatio)

        # 利用双线性差值法，计算U点处的像素值
        value = bilinear_insert(img, pt_U[0], pt_U[1])
        copy_img[j, i] = value

    return copy_img


# 大眼
@jit
def enlarge_eyes(image, face_landmark, strength=10):
    """
    image： 人像图片
    face_landmark: 人脸关键点
    radius: 眼睛放大范围半径
    strength：眼睛放大程度
    """
    img = image.copy()

    landmarks = face_landmark

    # 大眼调节参数
    scaleRatio = strength / 100

    # 小眼调节参数
    # scaleRatio =-1

    # 右眼
    index = [37, 38, 40, 41]
    pts_right_eyes = landmarks[index]
    crop_rect = cv2.boundingRect(pts_right_eyes)
    (x, y, w, h) = crop_rect
    pt_C_right = np.array([x + w / 2, y + h / 2], dtype=np.int32)

    r1 = np.sqrt(np.dot(pt_C_right - landmarks[36], pt_C_right - landmarks[36]))
    r2 = np.sqrt(np.dot(pt_C_right - landmarks[39], pt_C_right - landmarks[39]))
    R_right = 1.5 * np.max([r1, r2])

    # 左眼
    index = [43, 44, 45, 47]
    pts_left_eyes = landmarks[index]
    crop_rect = cv2.boundingRect(pts_left_eyes)
    (x, y, w, h) = crop_rect
    pt_C_left = np.array([x + w / 2, y + h / 2], dtype=np.int32)
    r1 = np.sqrt(np.dot(pt_C_left - landmarks[42], pt_C_left - landmarks[42]))
    r2 = np.sqrt(np.dot(pt_C_left - landmarks[46], pt_C_left - landmarks[46]))
    R_left = 1.5 * np.max([r1, r2])

    logging.info("processing  scaleRatio= %.2f" % (scaleRatio))

    # 大右眼
    img_bigeye = localScaleWap(img, pt_C_right, R_right, scaleRatio)

    # 大左眼
    img_bigeye = localScaleWap(img_bigeye, pt_C_left, R_left, scaleRatio)

    return img_bigeye


# 涂口红
@jit
def rouge(image, face_landmark, ruby=True):
    """
    image： 人像图片
    face_landmark: 人脸关键点
    ruby：是否需要深色口红
    """
    image_cp = image.copy()

    if ruby:
        rouge_color = (0, 0, 255)
    else:
        rouge_color = (0, 0, 200)

    points = face_landmark[48:68]

    hull = cv2.convexHull(points)
    cv2.drawContours(image, [hull], -1, rouge_color, -1)
    cv2.addWeighted(image, 0.2, image_cp, 1 - 0.1, 0, image_cp)
    return image_cp


Color_list = [
    1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 31, 33, 35, 37, 39,
    41, 43, 44, 46, 48, 50, 52, 53, 55, 57, 59, 60, 62, 64, 66, 67, 69, 71, 73, 74,
    76, 78, 79, 81, 83, 84, 86, 87, 89, 91, 92, 94, 95, 97, 99, 100, 102, 103, 105,
    106, 108, 109, 111, 112, 114, 115, 117, 118, 120, 121, 123, 124, 126, 127, 128,
    130, 131, 133, 134, 135, 137, 138, 139, 141, 142, 143, 145, 146, 147, 149, 150,
    151, 153, 154, 155, 156, 158, 159, 160, 161, 162, 164, 165, 166, 167, 168, 170,
    171, 172, 173, 174, 175, 176, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187,
    188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203,
    204, 205, 205, 206, 207, 208, 209, 210, 211, 211, 212, 213, 214, 215, 215, 216,
    217, 218, 219, 219, 220, 221, 222, 222, 223, 224, 224, 225, 226, 226, 227, 228,
    228, 229, 230, 230, 231, 232, 232, 233, 233, 234, 235, 235, 236, 236, 237, 237,
    238, 238, 239, 239, 240, 240, 241, 241, 242, 242, 243, 243, 244, 244, 244, 245,
    245, 246, 246, 246, 247, 247, 248, 248, 248, 249, 249, 249, 250, 250, 250, 250,
    251, 251, 251, 251, 252, 252, 252, 252, 253, 253, 253, 253, 253, 254, 254, 254,
    254, 254, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 256]


# 美白
# v1:磨皮程度
#@jit
def whitening(image, face_landmark, v1=3, v2=2):
    img = image.copy()

    white_ratio = v1 / 100

    v1 = v2 * 4 / 100


    # v2: 细节程度
    tmp1 = cv2.bilateralFilter(img, int(v1 * 5), v1 * 12.5, v1 * 12.5)
    # color_map = lambda x: Color_list[int(x)]
    # vfunc = np.vectorize(color_map)

    # tmp1 = cv2.subtract(tmp1, img)
    # tmp1 = cv2.add(tmp1, (10, 10, 10, 128))
    # tmp1 = cv2.GaussianBlur(tmp1, (2 * 2 - 1, 2 * 2 - 1), 0)
    #
    # tmp1 = cv2.add(img, tmp1)

    # tmp1 = vfunc(tmp1)
    dst = cv2.addWeighted(img, (1-white_ratio), tmp1, white_ratio, 0.0)


    return dst


def image_beautify(imgs, thin=0, enlarge=0, whiten=0, details=0, module=None):
    import time
    t_start = time.time()

    result = module.keypoint_detection(images=imgs)

    logging.info("face detect takes:" + str(time.time() - t_start))
    logging.info("faces:" + str(len(result[0]['data'])))

    img = deepcopy(imgs)

    do_thin_face = (thin > 0)
    do_enlarge_eyes = (enlarge > 0)
    do_whitening = (whiten > 0) or (details > 0)

    # 瘦脸
    if do_thin_face:
        for i in range(len(img)):
            for j in range(len(result[i]['data'])):
                img[i] = thin_face(img[i], np.array(result[i]['data'][j], dtype='int'), thin)

    # 放大双眼
    if do_enlarge_eyes:
        for i in range(len(img)):
            for j in range(len(result[i]['data'])):
                img[i] = enlarge_eyes(img[i], np.array(result[i]['data'][j], dtype='int'), enlarge)

    # 美白
    if do_whitening:
        for i in range(len(img)):
            for j in range(len(result[i]['data'])):
                img[i] = whitening(img[i], np.array(result[i]['data'][j], dtype='int'), whiten, details)

    logging.info("takes:" + str(time.time() - t_start))
    # 返回处理的对象数组
    return img

