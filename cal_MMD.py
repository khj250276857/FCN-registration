#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/20 14:27
# @Author  : Hongjian Kang
# @File    : cal_MMD.py

import tensorflow as tf

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    将源域和目标域数据转化为核矩阵
    :param source:  源域数据，n*len(x)
    :param target:  目标域数据， m*len(y)
    :param kernel_mul:
    :param kernel_num:  取不同高斯核的数量
    :param fix_sigma:   不同高斯核的sigma值
    :return:
    '''

    n_examples = source.shape[0] + target.shape[0]
    total = tf.concat([source, target], axis=0)
