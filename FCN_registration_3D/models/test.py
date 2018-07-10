#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/6 20:53
# @Author  : Hongjian Kang
# @File    : test.py

import tensorflow as tf
from FCN_registration_3D.models.utils import conv3d, conv3d_transpose, reg3d


class UNet(object):
    def __init__(self):
        name = 'unet'
        is_train = True
        self._name = name
        self._is_train = is_train
        self._reuse = None

    def __call__(self, x):
        batch_size = 10
        x = tf.placeholder(dtype='float32', shape=[batch_size, 64, 64, 64, 1])
            # encoding path
        x_1 = conv3d(x, 'Conv1', 32, 3, 2, 'SAME', True, tf.nn.relu, self._is_train)
        x_2 = tf.nn.avg_pool3d(x_1, [1, 3, 3, 3, 1], [1, 2, 2, 2, 1], 'SAME', name='pooling1')
        x_3 = conv3d(x_2, 'Conv2', 64, 3, 2, 'SAME', True, tf.nn.relu, self._is_train)
        x_4 = tf.nn.avg_pool3d(x_3, [1, 3, 3, 3, 1], [1, 2, 2, 2, 1], 'SAME', name='pooling2')
        x_5 = conv3d(x_4, 'Conv3', 128, 3, 2, 'SAME', True, tf.nn.relu, self._is_train)
        x_6 = conv3d_transpose(x_5, 'deconv1', 64, [batch_size, 4, 4, 4, 64], 3, 2, 'SAME', True, tf.nn.relu,
                               self._is_train)
        x_7 = conv3d(x_6, 'Conv4', 64, 3, 2, 'SAME', True, tf.nn.relu, self._is_train)
        x_8 = conv3d_transpose(x_7, 'deconv2', 32, [batch_size, 4, 4, 4, 32], 3, 2, 'SAME', True, tf.nn.relu,
                               self._is_train)

        # x_9,x_10,x_11 as regression layer, corresponding Reg1,Reg2 and Reg3 respectively
        x_9 = reg3d(x_8, 'Reg1', 2, 3, 1, 'SAME', self._is_train)
        x_10 = reg3d(x_7, 'Reg2', 2, 3, 1, 'SAME', self._is_train)
        x_11 = reg3d(x_5, 'Reg3', 2, 3, 1, 'SAME', self._is_train)

        print('x:', x)
        print('x_5:', x_5)
        print('x_7:', x_7)
        print('x_8:', x_8)
        print('x_9:', x_9)
        print('x_10:', x_10)
        print('x_11:', x_11)


UNet()(None)