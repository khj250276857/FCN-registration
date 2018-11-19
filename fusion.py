#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/15 14:49
# @Author  : Hongjian Kang
# @File    : fusion.py
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imsave
import os

x_path = r'C:\Users\khj\Desktop\实验结果\DenseNet_V1\batch_size=8,weight=200000,learning_rate=0.01\pt_2.png'
y_path = r'C:\Users\khj\Desktop\实验结果\DenseNet_V1\batch_size=8,weight=200000,learning_rate=0.01\ct_2.png'
z_path = r'C:\Users\khj\Desktop\实验结果\DenseNet_V1\batch_size=8,weight=200000,learning_rate=0.01\registrated_pt_2.png'
x_array = np.array(Image.open(x_path).convert('RGB'))
y_array = np.array(Image.open(y_path).convert('RGB'))
z_array = np.array(Image.open(z_path).convert('RGB'))

old_array = x_array[:, :, :]
new_array = z_array[:, :, :]
for i in range(x_array.shape[0]):
    for j in range(x_array.shape[1]):
        old_array[i, j][0] = x_array[i, j][0] * 0 + y_array[i, j][0] * 1
        new_array[i, j][0] = z_array[i, j][0] * 0 + y_array[i, j][0] * 1
        # old_array[i, j][0] = x_array[i, j][0]
        # new_array[i, j][0] = z_array[i, j][0]

imsave(os.path.join(r'C:\Users\khj\Desktop\slice data', 'fusion_old.png'), old_array)
imsave(os.path.join(r'C:\Users\khj\Desktop\slice data', 'fusion_new.png'), new_array)

# plt.figure('before')
# plt.imshow(old_array,cmap='jet')
# plt.figure('after')
# plt.imshow(new_array)
# plt.show()