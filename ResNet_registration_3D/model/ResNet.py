#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/9 18:11
# @Author  : Hongjian Kang
# @File    : ResNet.py

# ResNet-18 architecture

import tensorflow as tf
from ResNet_registration_3D.model.utils import conv3d, res_block_3D_with_ds, res_block_3D_without_ds, reg3d
from ResNet_registration_3D.model.spatial_transformer_3d import SpatialTransformer3D
from ResNet_registration_3D.model.utils import ncc_3d, grad_3d
import os
import pickle as pkl

class ResNet(object):
    def __init__(self, name: str, is_train: bool):
        self._name = name
        self._is_train = is_train
        self._reuse = None

    def __call__(self, x):
        # tf.reset_default_graph()
        with tf.variable_scope(self._name, reuse=self._reuse):
            batch_size = 4  # todo:change it

            x_1 = conv3d(x, 'Conv1', 32, 3, 1, 'SAME', True, tf.nn.relu, self._is_train)
            x_2 = tf.nn.max_pool3d(x_1, [1, 3, 3, 3, 1], [1, 2, 2, 2, 1], 'SAME', name='pooling1')      # size=10*32*32*32
            res_1 = res_block_3D_without_ds(x_2, 'ResBlock_1', 32, 3, 1, 'SAME', True, tf.nn.relu, self._is_train)
            res_2 = res_block_3D_without_ds(res_1, 'ResBlock_2', 32, 3, 1, 'SAME', True, tf.nn.relu, self._is_train)
            res_3 = res_block_3D_with_ds(res_2, 'ResBlock_3', 64, 3, 1, 'SAME', True, tf.nn.relu, self._is_train)     #size=10*16*16*16
            res_4 = res_block_3D_without_ds(res_3, 'ResBlock_4', 64, 3, 1, 'SAME', True, tf.nn.relu,  self._is_train)
            res_5 = res_block_3D_with_ds(res_4, 'ResBlock_5', 128, 3, 1, 'SAME', True, tf.nn.relu, self._is_train)     #size=10*8*8*8
            res_6 = res_block_3D_without_ds(res_5, 'ResBlock_6', 128, 3, 1, 'SAME', True, tf.nn.relu, self._is_train)
            res_7 = res_block_3D_with_ds(res_6, 'ResBlock_7', 256, 3, 1, 'SAME', True, tf.nn.relu, self._is_train)     #size=10*4*4*4
            res_8 = res_block_3D_without_ds(res_7, 'ResBlock_8', 256, 3, 1, 'SAME', True, tf.nn.relu, self._is_train)
            reg = reg3d(res_8, 'reg', 3, 3, 1, 'SAME', self._is_train)

        if self._reuse is None:
            self.var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._name)
            self.saver = tf.train.Saver(self.var_list)
            self._reuse = True
        return reg

    def save(self, session, checkpoint_path):
        self.saver.save(session, checkpoint_path)

    def restore(self, session, checkpoint_path):
        self.saver.restore(session, checkpoint_path)



class ResNetRegressor(object):
    def __init__(self, sess: tf.Session, is_train: bool, config: dict):
        # get trainNet parameters
        self._sess = sess
        _is_train = is_train
        _batch_size = config['batch_size']
        _img_height, _img_width, _img_depth = config["image_size"]
        self.x = tf.placeholder(dtype=tf.float32, shape=[_batch_size, _img_height, _img_width, _img_depth, 1])
        self.y = tf.placeholder(dtype=tf.float32, shape=[_batch_size, _img_height, _img_width, _img_depth, 1])
        self.xy = tf.concat([self.x, self.y], axis=4)  # [batch_size, img_height, img_width, _img_depth, 2]

        # construct Spatial Transformers
        self._resnet = ResNet('ResNet', is_train=_is_train)
        self.resnet_out = self._resnet(tf.transpose(self.xy, perm=[0, 3, 1, 2, 4]))

        self._v = tf.transpose(self.resnet_out, perm=[0, 2, 3, 1, 4])
        self.spatial_transformer = SpatialTransformer3D()
        self._z = self.spatial_transformer.transform(self.x, self._v)

        self.loss_ncc = -ncc_3d(self.y, self._z)
        self.loss_grad = grad_3d(self._v)
        self.loss = self.loss_ncc + self.loss_grad / 10000

        # construct trainNet step
        if _is_train:
            _optimizer = tf.train.AdamOptimizer(config['learning_rate'])
            _var_list = self._resnet.var_list
            self.train_step = _optimizer.minimize(self.loss, var_list=_var_list)

        # initialize all variables
        self._sess.run(tf.global_variables_initializer())

    def fit(self, batch_x, batch_y):
        _, loss, loss_ncc, loss_grad = self._sess.run(
            fetches=[self.train_step, self.loss, self.loss_ncc, self.loss_grad],
            feed_dict={self.x: batch_x, self.y: batch_y}
        )
        return loss, loss_ncc, loss_grad

    def deploy(self, batch_x, batch_y, save_path, patch_name_start_index=0):
        z = self._sess.run(self._z, feed_dict={self.x: batch_x, self.y: batch_y})
        loss, loss_ncc, loss_grad = self._sess.run(
            [self.loss, self.loss_ncc, self.loss_grad],
            feed_dict={self.x: batch_x, self.y: batch_y})
        if save_path is not None:
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
        return loss, loss_ncc, loss_grad


    def save(self, sess, save_folder: str):
        self._resnet.save(sess, os.path.join(save_folder, 'ResNet.ckpt'))

    def restore(self, sess, save_folder: str):
        self._resnet.restore(sess, os.path.join(save_folder, 'ResNet.ckpt'))
