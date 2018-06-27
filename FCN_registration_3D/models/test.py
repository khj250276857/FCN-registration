import tensorflow as tf
from models.utils import conv2d, conv2d_transpose, reg

class CNN(object):
    def __init__(self):
        #

        # todo : remove it
        name = "234"
        is_train = True
        # todo end

        self._name = name
        self._is_train = is_train
        self._reuse = None

    def __call__(self, x):
        x = tf.placeholder(dtype=tf.float32, shape=[10, 128, 128, 1])
        batch_size = 10
        x_1 = conv2d(x, 'Conv1', 32, 3, 2, 'SAME', True, tf.nn.relu, self._is_train)
        x_2 = tf.nn.avg_pool(x_1, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME', name='pooling1')
        x_3 = conv2d(x_2, 'Conv2', 64, 3, 2, 'SAME', True, tf.nn.relu, self._is_train)
        x_4 = tf.nn.avg_pool(x_3, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME', name='pooling2')
        x_5 = conv2d(x_4, 'Conv3', 128, 3, 2, 'SAME', True, tf.nn.relu, self._is_train)
        x_6 = conv2d_transpose(x_5, 'deconv1', 64, [batch_size, 8, 8, 64], 3, 2, 'SAME', True, tf.nn.relu, self._is_train)
        x_7 = conv2d(x_6, 'Conv4', 64, 3, 2, 'SAME', True, tf.nn.relu, self._is_train)
        x_8 = conv2d_transpose(x_7, 'deconv2', 32, [batch_size, 8, 8, 32], 3, 2, 'SAME', True, tf.nn.relu, self._is_train)

        # x_9,x_10,x_11 as regression layer
        x_9 = reg(x_8, 'Reg1', 2, 3, 1, 'SAME', self._is_train)
        x_10 = reg(x_7, 'Reg2', 2, 3, 1, 'SAME', self._is_train)
        x_11 = reg(x_5, 'Reg3', 2, 3, 1, 'SAME', self._is_train)

        # return x_9
        print('x_9: ', x_9)
        print('x_10: ', x_10)
        print('x_11: ', x_11)


CNN()(None)
