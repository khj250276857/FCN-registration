import tensorflow as tf
import numpy as np
from scipy.misc import imsave

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


# conv2d_transpose filter: A 4D Tensor[height, width, output_channels, in_channels].
def conv3d_transpose(x, name, dim, output_shape, k, s, p, bn, af, is_train):
    with tf.variable_scope(name):
        w = tf.get_variable('weight', [k, k, k, dim, x.get_shape()[-1]],
                            initializer=tf.truncated_normal_initializer(stddev=0.01))
        x = tf.nn.conv3d_transpose(x, w, output_shape, [1, s, s, s, 1], p)
        if bn:
            x = batch_norm(x, "bn", is_train=is_train)
        else:
            b = tf.get_variable('biases', [dim], initializer=tf.constant_initializer(0.))
            x += b
        if af:
            x = af(x)
    return x


# def reg(x, name, dim, k, s, p, is_train):
#     with tf.variable_scope(name):
#         w = tf.get_variable('weight', [k, k, k, x.get_shape()[-1], dim],
#                             initializer=tf.truncated_normal_initializer(stddev=0.01))
#         x = tf.nn.conv3d(x, w, [1, s, s, s, 1], p)
#     return x


def batch_norm(x, name, momentum=0.9, epsilon=1e-5, is_train=True):
    return tf.contrib.layers.batch_norm(x,
                                        decay=momentum,
                                        updates_collections=None,
                                        epsilon=epsilon,
                                        scale=True,
                                        is_training=is_train,
                                        scope=name)


def ncc_2d(x, y):
    mean_x = tf.reduce_mean(x, [1, 2, 3], keepdims=True)
    mean_y = tf.reduce_mean(y, [1, 2, 3], keepdims=True)
    mean_x2 = tf.reduce_mean(tf.square(x), [1, 2, 3], keepdims=True)
    mean_y2 = tf.reduce_mean(tf.square(y), [1, 2, 3], keepdims=True)
    stddev_x = tf.reduce_sum(tf.sqrt(mean_x2 - tf.square(mean_x)), [1, 2, 3], keepdims=True)
    stddev_y = tf.reduce_sum(tf.sqrt(mean_y2 - tf.square(mean_y)), [1, 2, 3], keepdims=True)
    return tf.reduce_mean((x - mean_x) * (y - mean_y) / (stddev_x * stddev_y))


def ncc_3d(x, y):
    mean_x = tf.reduce_mean(x, [1, 2, 3, 4], keepdims=True)
    mean_y = tf.reduce_mean(y, [1, 2, 3, 4], keepdims=True)
    mean_x2 = tf.reduce_mean(tf.square(x), [1, 2, 3, 4], keepdims=True)
    mean_y2 = tf.reduce_mean(tf.square(y), [1, 2, 3, 4], keepdims=True)
    stddev_x = tf.reduce_sum(tf.sqrt(mean_x2 - tf.square(mean_x)), [1, 2, 3, 4], keepdims=True)
    stddev_y = tf.reduce_sum(tf.sqrt(mean_y2 - tf.square(mean_y)), [1, 2, 3, 4], keepdims=True)
    return tf.reduce_mean((x - mean_x) * (y - mean_y) / (stddev_x * stddev_y))

def mse(x, y):
    return tf.reduce_mean(tf.square(x - y))


def grad_2d(v):
    """
        vectorized version of grad_xy
        :param deformation_field_matrix: 形变场矩阵（Tensor）
            shape: [batch_size, img_height, img_width, channels]
            typically, shape is [32, 8, 8, 2]
            dtype: float32
        :return: grad
        for a matrix
        [[3 4 5],
         [6 7 8],
         [9 1 2]]
        grad_x = reduce_sum(abs(
                [[4, 5],[7, 8], [1, 2]] - [[3, 4], [6, 7], [9, 1]]
            ))
        grad_y = reduce_sum(abs(
                [[6, 7, 8], [9, 1, 2]] - [[3, 4, 5], [6, 7, 8]]
            ))
        grad = grad_x + grad_y
        """
    img_height = v.shape[1]
    img_width = v.shape[2]
    grad_x = tf.reduce_sum(tf.abs(v[:, :, :img_width - 1, :] - v[:, :, 1:, :]))
    grad_y = tf.reduce_sum(tf.abs(v[:, :img_height - 1, :, :] - v[:, 1:, :, :]))
    return grad_x + grad_y


def grad_3d(v):
    """vectorized version of gradient against x, y, z axis"""
    img_height = v.shape[1]
    img_width = v.shape[2]
    img_depth = v.shape[3]
    grad_x = tf.reduce_sum(tf.abs(v[:, :, 1:, :, :] - v[:, :, :img_width - 1, :, :]))
    grad_y = tf.reduce_sum(tf.abs(v[:, 1:, :, :, :] - v[:, :img_height - 1, :, :, :]))
    grad_z = tf.reduce_sum(tf.abs(v[:, :, :, 1:, :] - v[:, :, :, :img_depth - 1, :]))
    return grad_x + grad_y + grad_z


def save_image_with_scale(path, arr):
    arr = np.clip(arr, 0., 1.)
    arr = arr * 255.
    arr = arr.astype(np.uint8)
    imsave(path, arr)





