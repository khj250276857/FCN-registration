#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/1 21:16
# @Author  : Hongjian Kang
# @File    : cal_IoU_in_whole_file.py

# 计算x、y、z三个文件夹中所有配准前后PET/CT图像肝脏检测框的交叠率，返回两个数组

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

def cal_IoU(a_array, b_array):
    #每个array中背景为0，检测框内部为1，返回两个图中检测框的重叠率（检测框的交集 / 并集）
    sum1, sum2 = 0, 0 #sum1是交集，sum2是并集
    for i in range(128):
        for j in range(128):
            if a_array[i, j] and b_array[i, j]:
                sum1 += 1
            if a_array[i, j] or b_array[i, j]:
                sum2 += 1
    return sum1 / sum2


def gen_array_box(pt_array, ct_array, registrated_pt_array):
    # 根据画有检测框的图像，生成背景为0、检测框内区域为1的图像
    rows, cols = 128, 128
    new_pt_array, new_ct_array, new_registrated_pt_array = np.zeros([rows, cols]), np.zeros([rows, cols]), np.zeros([rows, cols])
    for i in range(rows):
        for j in range(cols):
            if ct_array[i, j, 0] != ct_array[i, j, 1]:
                new_ct_array[i, j] = 1
            if pt_array[i, j, 0] != pt_array[i, j, 1]:
                new_pt_array[i, j] = 1
            if registrated_pt_array[i, j, 0] != registrated_pt_array[i, j, 1]:
                new_registrated_pt_array[i, j] = 1
    return new_pt_array, new_ct_array, new_registrated_pt_array

def main():
    pt_path = r'C:\Users\khj\Desktop\5_x'
    ct_path = r'C:\Users\khj\Desktop\5_y'
    registrated_pt_path = r'C:\Users\khj\Desktop\5_z'
    pt_images = [os.path.join(pt_path, _) for _ in os.listdir(pt_path)]
    ct_images = [os.path.join(ct_path, _) for _ in os.listdir(ct_path)]
    registrated_pt_images = [os.path.join(registrated_pt_path, _) for _ in os.listdir(registrated_pt_path)]

    original_result, registrated_result = [], []

    for i in range(len(pt_images)):
        pt_array = np.array(Image.open(pt_images[i]), dtype='float32')
        ct_array = np.array(Image.open(ct_images[i]), dtype='float32')
        registrated_pt_array = np.array(Image.open(registrated_pt_images[i]), dtype='float32')

        new_pt_array, new_ct_array, new_registrated_pt_array = gen_array_box(pt_array, ct_array, registrated_pt_array)
        original_result.append(cal_IoU(new_ct_array, new_pt_array))
        registrated_result.append(cal_IoU(new_ct_array, new_registrated_pt_array))

    print('original_avg: ', sum(original_result) / len(original_result))
    print('registrated_avg: ', sum(registrated_result) / len(registrated_result))


    # plt.figure('ct_bounding_box')
    # plt.imshow(new_ct_array, cmap='gray')
    # plt.figure('pt_bounding_box')
    # plt.imshow(new_ct_array, cmap='gray')
    # plt.show()

if __name__ == '__main__':
    main()