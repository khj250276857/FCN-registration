#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/18 16:52
# @Author  : Hongjian Kang
# @File    : pickle_to_slice.py

import pickle as pkl
import os
from scipy.misc import imsave


def save_2d_from_3d(pickle_path, save_path):  # pickle_path为pickle文件绝对路径
    print('Processing  {}'.format(pickle_path))
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # name = os.path.split(pickle_path)[-1].split('.')[0]
    with open(pickle_path, 'rb') as f:
        array = pkl.load(f)
    for i in range(array.shape[2]):
        imsave(os.path.join(os.path.join(save_path, '_{:>02}.png'.format(i+1))), array[:, :, i])



def main():
    x_path = r'C:\Users\khj\Desktop\running data(fcn, learning_rate=0.001,weight=100000,10000,10000)\validate\5_x.pkl'
    y_path = r'C:\Users\khj\Desktop\running data(fcn, learning_rate=0.001,weight=100000,10000,10000)\validate\5_y.pkl'
    z_path = r'C:\Users\khj\Desktop\running data(fcn, learning_rate=0.001,weight=100000,10000,10000)\validate\5_z1.pkl'

    save_path = r'C:\Users\khj\Desktop\slice data'
    x_save_path = os.path.join(save_path, os.path.split(x_path)[-1].split('.')[0])
    y_save_path = os.path.join(save_path, os.path.split(y_path)[-1].split('.')[0])
    z_save_path = os.path.join(save_path, os.path.split(z_path)[-1].split('.')[0])

    save_2d_from_3d(x_path, x_save_path)
    save_2d_from_3d(y_path, y_save_path)
    save_2d_from_3d(z_path, z_save_path)

if __name__ == '__main__':
    main()




