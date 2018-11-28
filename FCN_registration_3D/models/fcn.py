#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/6 20:18
# @Author  : Hongjian Kang
# @File    : fcn.py

import tensorflow as tf
from FCN_registration_3D.models.utils import conv3d, conv3d_transpose, reg3d
from FCN_registration_3D.models.spatial_transformer_3d import SpatialTransformer3D
from FCN_registration_3D.models.utils import ncc_3d, grad_3d
import os
import pickle as pkl

class FCN(object):
    def __init__(self, name: str, is_train: bool):
        self._name = name
        self._is_train = is_train
        self._reuse = None

    def __call__(self, x):
        with tf.variable_scope(self._name, reuse=self._reuse):
            batch_size = 2
            x_1 = conv3d(x, 'Conv1', 32, 3, 2, 'SAME', True, tf.nn.relu, self._is_train)
            x_2 = tf.nn.avg_pool3d(x_1, [1, 3, 3, 3, 1], [1, 2, 2, 2, 1], 'SAME', name='pooling1')
            x_3 = conv3d(x_2, 'Conv2', 64, 3, 2, 'SAME', True, tf.nn.relu, self._is_train)
            x_4 = tf.nn.avg_pool3d(x_3, [1, 3, 3, 3, 1], [1, 2, 2, 2, 1], 'SAME', name='pooling2')
            x_5 = conv3d(x_4, 'Conv3', 128, 3, 2, 'SAME', True, tf.nn.relu, self._is_train)
            x_6 = conv3d_transpose(x_5, 'deconv1', 64, [batch_size, 4, 4, 4, 64], 3, 2, 'SAME', True, tf.nn.relu, self._is_train)
            x_7 = conv3d(x_6, 'Conv4', 64, 3, 2, 'SAME', True, tf.nn.relu, self._is_train)
            x_8 = conv3d_transpose(x_7, 'deconv2', 32, [batch_size, 4, 4, 4, 32], 3, 2, 'SAME', True, tf.nn.relu, self._is_train)

            # x_9,x_10,x_11 as regression layer, corresponding Reg1,Reg2 and Reg3 respectively
            x_9 = reg3d(x_8, 'Reg1', 3, 3, 1, 'SAME', self._is_train)
            x_10 = reg3d(x_7, 'Reg2', 3, 3, 1, 'SAME', self._is_train)
            x_11 = reg3d(x_5, 'Reg3', 3, 3, 1, 'SAME', self._is_train)
        if self._reuse is None:
            self.var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._name)
            self.saver = tf.train.Saver(self.var_list)
            self._reuse = True
        return x_9, x_10, x_11

    def save(self, session, checkpoint_path):
        self.saver.save(session, checkpoint_path)

    def restore(self, session, checkpoint_path):
        self.saver.restore(session, checkpoint_path)


class fcnRegressor(object):
    def __init__(self, sess: tf.Session, is_train: bool, config: dict):
        # get trainNet parameters
        self._sess = sess
        _is_train = is_train
        _batch_size = config['batch_size']
        _img_height, _img_width, _img_depth = config["image_size"]
        self.x = tf.placeholder(dtype=tf.float32, shape=[_batch_size, _img_height, _img_width, _img_depth, 1])
        self.y = tf.placeholder(dtype=tf.float32, shape=[_batch_size, _img_height, _img_width, _img_depth, 1])
        self.xy = tf.concat([self.x, self.y], axis=4)    # [batch_size, img_height, img_width, _img_depth, 2]

        # construct Spatial Transformers
        self._fcn = FCN('FCN', is_train=_is_train)
        self.fcn_out = self._fcn(tf.transpose(self.xy, perm=[0, 3, 1, 2, 4]))

        # todo: remove it
        self._v1 = tf.transpose(self.fcn_out[0], perm=[0, 2, 3, 1, 4])
        self._v2 = tf.transpose(self.fcn_out[1], perm=[0, 2, 3, 1, 4])
        self._v3 = tf.transpose(self.fcn_out[2], perm=[0, 2, 3, 1, 4])
        # self._v1 = self.fcn_out[0]
        # self._v2 = self.fcn_out[1]
        # self._v3 = self.fcn_out[2]

        self.spatial_transformer = SpatialTransformer3D()
        self._z1 = self.spatial_transformer.transform(self.x, self._v1)
        self._z2 = self.spatial_transformer.transform(self.x, self._v2)
        self._z3 = self.spatial_transformer.transform(self.x, self._v3)

        # calculate loss
        self.loss1 = -ncc_3d(self.y, self._z1)
        self.loss2 = -ncc_3d(self.y, self._z2)
        self.loss3 = -ncc_3d(self.y, self._z3)
        self.loss4 = grad_3d(self._v1)
        self.loss5 = grad_3d(self._v2)
        self.loss6 = grad_3d(self._v3)
        self.loss = (self.loss1 + self.loss4 / 100000) + 0.6 * (self.loss2 + self.loss5 / 5000) + 0.3 * (self.loss3 + self.loss6 / 5000)

        # construct trainNet step
        if _is_train:
            _optimizer = tf.train.AdamOptimizer(config['learning_rate'])
            _var_list = self._fcn.var_list
            self.train_step = _optimizer.minimize(self.loss, var_list=_var_list)

        # initialize all variables
        self._sess.run(tf.global_variables_initializer())

    def fit(self, batch_x, batch_y):
        _, loss, loss1, loss2, loss3, loss4, loss5, loss6 = self._sess.run(
            fetches=[self.train_step, self.loss, self.loss1, self.loss2, self.loss3, self.loss4, self.loss5, self.loss6],
            feed_dict={self.x: batch_x, self.y: batch_y}
        )
        return loss, loss1, loss2, loss3, loss4, loss5, loss6

    def deploy(self, batch_x, batch_y, save_path, patch_name_start_index=0):
        z1, z2, z3 = self._sess.run([self._z1, self._z2, self._z3], feed_dict={self.x: batch_x, self.y: batch_y})
        loss, loss_1, loss_2, loss_3, loss_4, loss_5, loss_6 = self._sess.run([self.loss, self.loss1, self.loss2, self.loss3, self.loss4, self.loss5, self.loss6],
                                                      feed_dict={self.x: batch_x, self.y: batch_y})
        if save_path is not None:
            for i in range(z1.shape[0]):
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
                    obj=z1[i, :, :, :, 0],
                    file=open(os.path.join(save_path, '{}_z1.pkl'.format(_index)), 'wb')
                )
                pkl.dump(
                    obj=z2[i, :, :, :, 0],
                    file=open(os.path.join(save_path, '{}_z2.pkl'.format(_index)), 'wb')
                )
                pkl.dump(
                    obj=z3[i, :, :, :, 0],
                    file=open(os.path.join(save_path, '{}_z3.pkl'.format(_index)), 'wb')
                )
        return loss, loss_1, loss_2, loss_3, loss_4, loss_5, loss_6

    def save(self, sess, save_folder: str):
        self._fcn.save(sess, os.path.join(save_folder, 'FCN.ckpt'))

    def restore(self, sess, save_folder: str):
        self._fcn.restore(sess, os.path.join(save_folder, 'FCN.ckpt'))