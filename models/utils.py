import tensorflow as tf
import numpy as np
from scipy.misc import imsave

def conv2d(x, name, dim, k, s, p, bn, af, is_train):
    with tf.variable_scope(name):
        w = tf.get_variable('weight', [k, k, x.get_shape()[-1], dim],
                            initializer=tf.truncated_normal_initializer(stddev=0.01))
        x = tf.nn.conv2d(x, w, [1, s, s, 1], p)
        if bn:
            x = batch_norm(x, "bn", is_train=is_train)
        else:
            b = tf.get_variable('biases', [dim], initializer=tf.constant_initializer(0.))
            x += b
        if af:
            x = af(x)
    return x


# conv2d_transpose filter: A 4D Tensor[height, width, output_channels, in_channels].
def conv2d_transpose(x, name, dim, output_shape, k, s, p, bn, af, is_train):
    with tf.variable_scope(name):
        w = tf.get_variable('weight', [k, k, dim, x.get_shape()[-1]],
                            initializer=tf.truncated_normal_initializer(stddev=0.01))
        # output_shape = tf.constant([10, 8, 8, 64])
        x = tf.nn.conv2d_transpose(x, w, output_shape, [1, s, s, 1], p)
        if bn:
            x = batch_norm(x, "bn", is_train=is_train)
        else:
            b = tf.get_variable('biases', [dim], initializer=tf.constant_initializer(0.))
            x += b
        if af:
            x = af(x)
    return x


def reg(x, name, dim, k, s, p, is_train):
    with tf.variable_scope(name):
        w = tf.get_variable('weight', [k, k, x.get_shape()[-1], dim],
                            initializer=tf.truncated_normal_initializer(stddev=0.01))
        x = tf.nn.conv2d(x, w, [1, s, s, 1], p)
    return x


def batch_norm(x, name, momentum=0.9, epsilon=1e-5, is_train=True):
    return tf.contrib.layers.batch_norm(x,
                                        decay=momentum,
                                        updates_collections=None,
                                        epsilon=epsilon,
                                        scale=True,
                                        is_training=is_train,
                                        scope=name)


def ncc(x, y):
    mean_x = tf.reduce_mean(x, [1, 2, 3], keepdims=True)
    mean_y = tf.reduce_mean(y, [1, 2, 3], keepdims=True)
    mean_x2 = tf.reduce_mean(tf.square(x), [1, 2, 3], keepdims=True)
    mean_y2 = tf.reduce_mean(tf.square(y), [1, 2, 3], keepdims=True)
    stddev_x = tf.reduce_sum(tf.sqrt(mean_x2 - tf.square(mean_x)), [1, 2, 3], keepdims=True)
    stddev_y = tf.reduce_sum(tf.sqrt(mean_y2 - tf.square(mean_y)), [1, 2, 3], keepdims=True)
    return tf.reduce_mean((x - mean_x) * (y - mean_y) / (stddev_x * stddev_y))


def mse(x, y):
    return tf.reduce_mean(tf.square(x - y))


def grad(v):
    num_batch = v.shape[0]
    height = v.shape[1]
    width = v.shape[2]
    channels = v.shape[3]

    grad_x = 0
    grad_y = 0
    for num in range(num_batch):
        v_x = v[num, :, :, 0]
        v_y = v[num, :, :, 1]
        for i in range(1, height-1):
            for j in range(1, width-1):
                grad_x_temp = abs(v_x[i, j-1] - v_x[i, j]) + abs(v_x[i, j] - v_x[i, j+1]) + abs(v_x[i-1, j] - v_x[i, j]) + abs(v_x[i, j] - v_x[i+1, j])
                grad_y_temp = abs(v_y[i, j-1] - v_y[i, j]) + abs(v_y[i, j] - v_y[i, j+1]) + abs(v_y[i-1, j] - v_y[i, j]) + abs(v_y[i, j] - v_y[i+1, j])
                grad_x += grad_x_temp
                grad_y += grad_y_temp
    grad_result = grad_x + grad_y
    return grad_result


def save_image_with_scale(path, arr):
    arr = np.clip(arr, 0., 1.)
    arr = arr * 255.
    arr = arr.astype(np.uint8)
    imsave(path, arr)





