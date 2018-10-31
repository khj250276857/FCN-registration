#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/22 14:10
# @Author  : Hongjian Kang
# @File    : 2D_slices_to_3D_array.py

# 将文件中的图像依次读入3D array中，并显示冠状面
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def main():
    file_path = r'C:\Users\khj\Desktop\slice data\2_y'
    image_names = [os.path.join(file_path, _) for _ in os.listdir(file_path)]
    image_array = np.array([np.array(Image.open(image_name)) for image_name in image_names], dtype='float32')
    target_image = image_array[32, :, :]

    plt.figure('target_image')
    plt.imshow(target_image, cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()