#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/6 20:53
# @Author  : Hongjian Kang
# @File    : test.py

import tensorflow as tf
from ResNet_registration_3D.model.utils import conv3d, res_block_3D_with_ds, res_block_3D_without_ds, reg3d


class ResNet(object):
    def __init__(self):
        name = 'resnet'
        is_train = True
        self._name = name
        self._is_train = is_train
        self._reuse = None

    def __call__(self, x):
        batch_size = 10
        x = tf.placeholder(dtype='float32', shape=[batch_size, 64, 64, 64, 2])

        x_1 = conv3d(x, 'Conv1', 64, 5, 1, 'SAME', True, tf.nn.relu, self._is_train)
        x_2 = tf.nn.max_pool3d(x_1, [1, 3, 3, 3, 1], [1, 2, 2, 2, 1], 'SAME', name='pooling1')  # size=10*32*32*32
        res_1 = res_block_3D_without_ds(x_2, 'ResBlock_1', 64, 3, 1, 'SAME', True, tf.nn.relu, self._is_train)
        res_2 = res_block_3D_without_ds(res_1, 'ResBlock_2', 64, 3, 1, 'SAME', True, tf.nn.relu, self._is_train)
        res_3 = res_block_3D_with_ds(res_2, 'ResBlock_3', 128, 3, 1, 'SAME', True, tf.nn.relu,
                                     self._is_train)  # size=10*16*16*16
        res_4 = res_block_3D_without_ds(res_3, 'ResBlock_4', 128, 3, 1, 'SAME', True, tf.nn.relu, self._is_train)
        res_5 = res_block_3D_with_ds(res_4, 'ResBlock_5', 256, 3, 1, 'SAME', True, tf.nn.relu,
                                     self._is_train)  # size=10*8*8*8
        res_6 = res_block_3D_without_ds(res_5, 'ResBlock_6', 256, 3, 1, 'SAME', True, tf.nn.relu, self._is_train)
        res_7 = res_block_3D_with_ds(res_6, 'ResBlock_7', 512, 3, 1, 'SAME', True, tf.nn.relu,
                                     self._is_train)  # size=10*4*4*4
        res_8 = res_block_3D_without_ds(res_7, 'ResBlock_8', 512, 3, 1, 'SAME', True, tf.nn.relu, self._is_train)
        reg = reg3d(res_8, 'reg', 3, 3, 1, 'SAME', self._is_train)

        print('x_1:', x_1)
        print('x_2:', x_2)
        print('res_1:', res_1)
        print('res_2:', res_2)
        print('res_3:', res_3)
        print('res_4:', res_4)
        print('res_5:', res_5)
        print('res_6:', res_6)
        print('res_7:', res_7)
        print('res_8:', res_8)
        print('reg:', reg)

ResNet()(None)