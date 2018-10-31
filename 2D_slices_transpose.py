#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/25 15:44
# @Author  : Hongjian Kang
# @File    : 2D_slices_transpose.py

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.misc import imsave

def gen_3D_array(file_path):    # 将文件中的2D图像生成3D array
    image_names = [os.path.join(file_path, _) for _ in os.listdir(file_path)]
    image_array = np.array([np.array(Image.open(image_name)) for image_name in image_names], dtype='float32')
    return image_array

def main():
    workspace = r'C:\Users\khj\Desktop\slice data'
    x_path = os.path.join(workspace, '8_x')
    y_path = os.path.join(workspace, '8_y')
    z_path = os.path.join(workspace, '8_z')
    pt_array = gen_3D_array(x_path)
    ct_array = gen_3D_array(y_path)
    registrated_pt_array = gen_3D_array(z_path)

    # imsave(os.path.join(workspace, 'pt_1.png'), pt_array[:, 64, :])
    # imsave(os.path.join(workspace, 'ct_1.png'), ct_array[:, 64, :])
    # imsave(os.path.join(workspace, 'regstrated_pt_1.png'), registrated_pt_array[:, 64, :])

    plt.figure('pt')
    plt.imshow(pt_array[:, 64, :], cmap='gray')
    plt.figure('ct')
    plt.imshow(ct_array[:, 64, :], cmap='gray')
    plt.figure('registrated_pt')
    plt.imshow(registrated_pt_array[:, 64, :], cmap='gray')

    plt.show()

if __name__ == '__main__':
    main()
