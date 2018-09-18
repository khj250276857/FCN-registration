#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/15 21:04
# @Author  : Hongjian Kang
# @File    : utils.py

import tensorflow as tf


def RDB(x, name, dim, k, s, p, bn, af, is_train):
    with tf.variable_scope(name):
        pass



def res_block_3D_with_ds(x, name, dim, k, s, p, bn, af, is_train):     # ds:downsampling
    with tf.variable_scope(name):
        w1 = tf.get_variable('weight1', [k, k, k, x.get_shape()[-1], dim],
                                initializer=tf.truncated_normal_initializer(stddev=0.01))

        x = tf.nn.conv3d(x, w1, [1, 2, 2, 2, 1], p)
        if bn:
            x = batch_norm(x, "bn1", is_train=is_train)
        else:
            b = tf.get_variable('biases1', [dim], initializer=tf.constant_initializer(0.))
            x += b
        if af:
            x = af(x)

        w2 = tf.get_variable('weight2', [k, k, k, x.get_shape()[-1], dim],
                             initializer=tf.truncated_normal_initializer(stddev=0.01))
        x = tf.nn.conv3d(x, w2, [1, s, s, s, 1], p)
        if bn:
            x = batch_norm(x, "bn2", is_train=is_train)
        else:
            b = tf.get_variable('biases2', [dim], initializer=tf.constant_initializer(0.))
            x += b
        if af:
            x = af(x)

    return x


def res_block_3D_without_ds(x, name, dim, k, s, p, bn, af, is_train):
    with tf.variable_scope(name):
        x_shortcut = x
        w1 = tf.get_variable('weight1', [k, k, k, x.get_shape()[-1], dim],
                                initializer=tf.truncated_normal_initializer(stddev=0.01))

        x = tf.nn.conv3d(x, w1, [1, s, s, s, 1], p)
        if bn:
            x = batch_norm(x, "bn1", is_train=is_train)
        else:
            b = tf.get_variable('biases1', [dim], initializer=tf.constant_initializer(0.))
            x += b
        if af:
            x = af(x)

        w2 = tf.get_variable('weight2', [k, k, k, x.get_shape()[-1], dim],
                            initializer=tf.truncated_normal_initializer(stddev=0.01))
        x = tf.nn.conv3d(x, w2, [1, s, s, s, 1], p)
        if bn:
            x = batch_norm(x, "bn2", is_train=is_train)
        else:
            b = tf.get_variable('biases2', [dim], initializer=tf.constant_initializer(0.))
            x += b

        add = tf.add(x, x_shortcut)
        add_result = tf.nn.relu(add)

    return add_result





def conv3d(x, name, dim, k, s, p, bn, af, is_train):
    with tf.variable_scope(name):
        w = tf.get_variable('weight', [k, k, k, x.get_shape()[-1], dim],
                            initializer=tf.truncated_normal_initializer(stddev=0.01))
        x = tf.nn.conv3d(x, w, [1, s, s, s, 1], p)
        if bn:
            x = batch_norm(x, "bn", is_train=is_train)
        else:
            b = tf.get_variable('biases', [dim], initializer=tf.constant_initializer(0.))
            x += b
        if af:
            x = af(x)
    return x


def reg3d(x, name, dim, k, s, p, is_train):
    with tf.variable_scope(name):
        w = tf.get_variable('weight', [k, k, k, x.get_shape()[-1], dim],
                            initializer=tf.truncated_normal_initializer(stddev=0.01))
        x = tf.nn.conv3d(x, w, [1, s, s, s, 1], p)
    return x



def batch_norm(x, name, momentum=0.9, epsilon=1e-5, is_train=True):
    return tf.contrib.layers.batch_norm(x,
                                        decay=momentum,
                                        updates_collections=None,
                                        epsilon=epsilon,
                                        scale=True,
                                        is_training=is_train,
                                        scope=name)


def ncc_3d(x, y):
    mean_x = tf.reduce_mean(x, [1, 2, 3, 4], keepdims=True)
    mean_y = tf.reduce_mean(y, [1, 2, 3, 4], keepdims=True)
    mean_x2 = tf.reduce_mean(tf.square(x), [1, 2, 3, 4], keepdims=True)
    mean_y2 = tf.reduce_mean(tf.square(y), [1, 2, 3, 4], keepdims=True)
    stddev_x = tf.reduce_sum(tf.sqrt(mean_x2 - tf.square(mean_x)), [1, 2, 3, 4], keepdims=True)
    stddev_y = tf.reduce_sum(tf.sqrt(mean_y2 - tf.square(mean_y)), [1, 2, 3, 4], keepdims=True)
    return tf.reduce_mean((x - mean_x) * (y - mean_y) / (stddev_x * stddev_y))

def grad_3d(v):
    """vectorized version of gradient against x, y, z axis"""
    img_height = v.shape[1]
    img_width = v.shape[2]
    img_depth = v.shape[3]
    grad_x = tf.reduce_sum(tf.abs(v[:, :, 1:, :, :] - v[:, :, :img_width - 1, :, :]))
    grad_y = tf.reduce_sum(tf.abs(v[:, 1:, :, :, :] - v[:, :img_height - 1, :, :, :]))
    grad_z = tf.reduce_sum(tf.abs(v[:, :, :, 1:, :] - v[:, :, :, :img_depth - 1, :]))
    return grad_x + grad_y + grad_z