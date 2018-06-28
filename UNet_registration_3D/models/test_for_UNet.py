#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/28 16:44
# @Author  : Hongjian Kang
# @File    : test_for_UNet.py

import tensorflow as tf
from FCN_registration_3D.models.utils import conv3d, conv3d_transpose, reg


class UNet(object):
    def __init__(self):
        name = 'unet'
        is_train = True
        self._name = name
        self._is_train = is_train
        self._reuse = None

    def __call__(self, x):
        batch_size = 10
        x = tf.placeholder(dtype='float32', shape=[batch_size, 128, 128, 128, 1])
            # encoding path
        x_1 = conv3d(x, 'Conv1', 16, 3, 1, 'SAME', True, tf.nn.leaky_relu, self._is_train)
        x_2 = conv3d(x_1, 'Conv2', 32, 3, 2, 'SAME', True, tf.nn.leaky_relu, self._is_train)
        x_3 = conv3d(x_2, 'Conv3', 32, 3, 2, 'SAME', True, tf.nn.leaky_relu, self._is_train)
        x_4 = conv3d(x_3, 'Conv4', 32, 3, 2, 'SAME', True, tf.nn.leaky_relu, self._is_train)
        x_5 = conv3d(x_4, 'Conv5', 32, 3, 2, 'SAME', True, tf.nn.leaky_relu, self._is_train)
        # decoding path
        x_6 = conv3d_transpose(x_5, 'Deconv1', 32, [10, 16, 16, 16, 32], 3, 2, 'SAME', True, tf.nn.leaky_relu, self._is_train)
        x_6_out = tf.concat([x_6, x_4], axis=4)
        x_7 = conv3d_transpose(x_6_out, 'Deconv2', 32, [10, 32, 32, 32, 32], 3, 2, 'SAME', True, tf.nn.leaky_relu, self._is_train)
        x_7_out = tf.concat([x_7, x_3], axis=4)
        x_8 = conv3d_transpose(x_7_out, 'Deconv3', 32, [10, 64, 64, 64, 32], 3, 2, 'SAME', True, tf.nn.leaky_relu, self._is_train)
        x_8_out = tf.concat([x_8, x_2], axis=4)
        x_9 = conv3d(x_8_out, 'Conv6', 32, 3, 1, 'SAME', True, tf.nn.leaky_relu, self._is_train)
        x_10 = conv3d_transpose(x_9, 'Deconv4', 16, [10, 128, 128, 128, 16], 3, 2, 'SAME', True, tf.nn.leaky_relu, self._is_train)
        x_10_out = tf.concat([x_10, x_1], axis=4)
        x_11 = conv3d(x_10_out, 'Conv7', 16, 3, 1, 'SAME', True, tf.nn.leaky_relu, self._is_train)
        x_12 = conv3d(x_11, 'Conv8', 3, 3, 1, 'SAME', True, tf.nn.leaky_relu, self._is_train)

        print('x:', x)
        print('x_1:', x_1)
        print('x_2:', x_2)
        print('x_3:', x_3)
        print('x_4:', x_4)
        print('x_5:', x_5)
        print('x_6:', x_6)
        print('x_7:', x_7)
        print('x_8:', x_8)
        print('x_9:', x_9)
        print('x_10:', x_10)
        print('x_11:', x_11)
        print('x_12:', x_12)
        print('/n')
        print('x_6_out:', x_6_out)
        print('x_7_out:', x_7_out)
        print('x_8_out:', x_8_out)
        print('x_10_out:', x_10_out)

UNet()(None)
