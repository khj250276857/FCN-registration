#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/6 20:21
# @Author  : Hongjian Kang
# @File    : bicubic_interpolation_3d.py

import tensorflow as tf

def interpolate_3d(arr, n, h, w, d, c, H, W, D):
    """
    interpolate 3d version
    :param arr: array
    :param n: batch_size
    :param h: img_height
    :param w: img_width
    :param d: img_depth
    :param c: img_channel
    :param H: new_img_height
    :param W: new_img_width
    :param D: new_img_depth
    :return:interpolated arr
    """
    arr = tf.reshape(arr, [n, h, w, d * c])
    arr = tf.image.resize_bicubic(arr, [H, W], True)  # [n, h, w, d, c] -> [n, H, W, d, c]
    arr = tf.reshape(arr, [n, H * W, d, c])
    arr = tf.image.resize_bicubic(arr, [H * W, D], True)  # [n, H, W, d, c] -> [n, H, W, D, c]
    return tf.reshape(arr, [n, H, W, D, c])