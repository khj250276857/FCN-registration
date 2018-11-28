#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/28 13:59
# @Author  : Hongjian Kang
# @File    : unet.py

import tensorflow as tf
from UNet_registration_3D.models.utils import conv3d, conv3d_transpose
from UNet_registration_3D.models.spatial_transformer_3d import SpatialTransformer3D
from UNet_registration_3D.models.utils import ncc_3d, save_image_with_scale, grad_3d
import pickle as pkl
import os

class UNet(object):
    def __init__(self, name: str, is_train: bool):
        self._name = name
        self._is_train = is_train
        self._reuse = None

    def __call__(self, x):
        with tf.variable_scope(self._name, reuse=self._reuse):
            batch_size = 2
            # encoding path
            x_1 = conv3d(x, 'Conv1', 16, 3, 1, 'SAME', True, tf.nn.leaky_relu, self._is_train)
            x_2 = conv3d(x_1, 'Conv2', 32, 3, 2, 'SAME', True, tf.nn.leaky_relu, self._is_train)
            x_3 = conv3d(x_2, 'Conv3', 32, 3, 2, 'SAME', True, tf.nn.leaky_relu, self._is_train)
            x_4 = conv3d(x_3, 'Conv4', 32, 3, 2, 'SAME', True, tf.nn.leaky_relu, self._is_train)
            x_5 = conv3d(x_4, 'Conv5', 32, 3, 2, 'SAME', True, tf.nn.leaky_relu, self._is_train)
            # decoding path
            x_6 = conv3d_transpose(x_5, 'Deconv1', 32, [batch_size, 8, 8, 8, 32], 3, 2, 'SAME', True, tf.nn.leaky_relu, self._is_train)
            x_6_out = tf.concat([x_6, x_4], axis=4, name='CA1')
            x_7 = conv3d_transpose(x_6_out, 'Deconv2', 32, [batch_size, 16, 16, 16, 32], 3, 2, 'SAME', True, tf.nn.leaky_relu, self._is_train)
            x_7_out = tf.concat([x_7, x_3], axis=4, name='CA2')
            x_8 = conv3d_transpose(x_7_out, 'Deconv3', 32, [batch_size, 32, 32, 32, 32], 3, 2, 'SAME', True, tf.nn.leaky_relu, self._is_train)
            x_8_out = tf.concat([x_8, x_2], axis=4, name='CA3')
            x_9 = conv3d(x_8_out, 'Conv6', 32, 3, 1, 'SAME', True, tf.nn.leaky_relu, self._is_train)
            x_10 = conv3d_transpose(x_9, 'Deconv4', 16, [batch_size, 64, 64, 64, 16], 3, 2, 'SAME', True, tf.nn.leaky_relu, self._is_train)
            x_10_out = tf.concat([x_10, x_1], axis=4, name='CA4')
            x_11 = conv3d(x_10_out, 'Conv7', 16, 3, 1, 'SAME', True, tf.nn.leaky_relu, self._is_train)
            x_12 = conv3d(x_11, 'Conv8', 3, 3, 1, 'SAME', True, tf.nn.leaky_relu, self._is_train)
        if self._reuse is None:
            self.var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._name)
            self.saver = tf.train.Saver(self.var_list)
            self._reuse = True
        return x_12

    def save(self, session, checkpoint_path):
        self.saver.save(session, checkpoint_path)

    def restore(self, session, checkpoint_path):
        self.saver.restore(session, checkpoint_path)


class unetRegressor(object):
    def __init__(self, sess: tf.Session, is_train: bool, config: dict):
        self._sess = sess
        _is_train = is_train
        _batch_size = config['batch_size']
        _img_depth, _img_height, _img_width = config["image_size"]
        self.x = tf.placeholder(dtype=tf.float32, shape=[_batch_size, _img_height, _img_width, _img_depth, 1])
        self.y = tf.placeholder(dtype=tf.float32, shape=[_batch_size, _img_height, _img_width, _img_depth, 1])
        self.xy = tf.concat([self.x, self.y], axis=4)  # [batch_size, img_height, img_width, img_depth, 2]

        # construct Spatial Transformers
        self._unet = UNet('UNet', is_train=_is_train)
        self.unet_out = self._unet(tf.transpose(self.xy, perm=[0, 3, 1, 2, 4]))
        self.v = tf.transpose(self.unet_out, perm=[0, 2, 3, 1, 4])

        # todo: reedit it
        self.spatial_transformer = SpatialTransformer3D()
        self.z = self.spatial_transformer.transform(self.x, self.v)

        # calculate loss
        self.loss1 = -ncc_3d(self.y, self.z)
        self.loss2 = grad_3d(self.v)
        self.loss = self.loss1 + self.loss2 / 50000000

        # construct trainNet step
        if _is_train:
            _optimizer = tf.train.AdamOptimizer(config['learning_rate'])
            _var_list = self._unet.var_list
            self.train_step = _optimizer.minimize(self.loss, var_list=_var_list)

        # # initialize all variables
        self._sess.run(tf.global_variables_initializer())

    def fit(self, batch_x, batch_y):
        _, loss, loss1, loss2 = self._sess.run(
            fetches=[self.train_step, self.loss, self.loss1, self.loss2],
            feed_dict={self.x: batch_x, self.y: batch_y}
        )
        return loss, loss1, loss2

    def deploy(self, batch_x, batch_y, save_path, patch_name_start_index=0):
        z, loss, loss1, loss2 = self._sess.run(
            fetches=[self.z, self.loss, self.loss1, self.loss2],
            feed_dict={self.x: batch_x, self.y: batch_y}
        )

        if save_path is not None:
            print('z.shape = {}'.format(z.shape))
            print('batch_x.shape = {}'.format(batch_x.shape))
            for i in range(z.shape[0]):
                _index = patch_name_start_index + i + 1
                pkl.dump(
                    obj=batch_x[i, :, :, :, 0],
                    file=open(os.path.join(save_path, '{}_x.pkl'.format(_index)), 'wb')
                )
                pkl.dump(
                    obj=batch_y[i, :, :, :, 0],
                    file=open(os.path.join(save_path, '{}_y.pkl'.format(_index)), 'wb')
                )
                pkl.dump(
                    obj=z[i, :, :, :, 0],
                    file=open(os.path.join(save_path, '{}_z.pkl'.format(_index)), 'wb')
                )
                # todo: return loss2
        return loss, loss1, loss2

    def save(self, sess, save_folder: str):
        self._unet.save(sess, os.path.join(save_folder, 'UNet.ckpt'))

    def restore(self, sess, save_folder: str):
        self._unet.restore(sess, os.path.join(save_folder, 'UNet.ckpt'))