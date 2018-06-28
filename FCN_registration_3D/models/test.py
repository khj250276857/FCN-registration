import tensorflow as tf
from FCN_registration_3D.models.utils import conv3d, conv3d_transpose, reg

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
        batch_size = 10
        x = tf.placeholder(dtype=tf.float32, shape=[batch_size, 128, 128, 128, 1])

        x_1 = conv3d(x, 'Conv1', 32, 3, 2, 'SAME', True, tf.nn.relu, self._is_train)
        x_2 = tf.nn.avg_pool3d(x_1, [1, 3, 3, 3, 1], [1, 2, 2, 2, 1], 'SAME', name='pooling1')
        x_3 = conv3d(x_2, 'Conv2', 64, 3, 2, 'SAME', True, tf.nn.relu, self._is_train)
        x_4 = tf.nn.avg_pool3d(x_3, [1, 3, 3, 3, 1], [1, 2, 2, 2, 1], 'SAME', name='pooling2')
        x_5 = conv3d(x_4, 'Conv3', 128, 3, 2, 'SAME', True, tf.nn.relu, self._is_train)
        x_6 = conv3d_transpose(x_5, 'deconv1', 64, [batch_size, 8, 8, 8, 64], 3, 2, 'SAME', True, tf.nn.relu, self._is_train)
        x_7 = conv3d(x_6, 'Conv4', 64, 3, 2, 'SAME', True, tf.nn.relu, self._is_train)
        x_8 = conv3d_transpose(x_7, 'deconv2', 32, [batch_size, 8, 8, 8, 32], 3, 2, 'SAME', True, tf.nn.relu, self._is_train)

        # x_9,x_10,x_11 as regression layer
        x_9 = reg(x_8, 'Reg1', 3, 3, 1, 'SAME', self._is_train)
        x_10 = reg(x_7, 'Reg2', 3, 3, 1, 'SAME', self._is_train)
        x_11 = reg(x_5, 'Reg3', 3, 3, 1, 'SAME', self._is_train)

        # return x_9
        print('x_9: ', x_9)
        print('x_10: ', x_10)
        print('x_11: ', x_11)


CNN()(None)
