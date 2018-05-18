import tensorflow as tf
from models.utils import conv2d
from models.utils import conv2d_transpose

class FCN(object):
    def __init__(self, name: str, is_train: bool):
        self._name = name
        self._is_train = is_train
        self._reuse = None

    def __call__(self, x):
        with tf.variable_scope(self._name, reuse=self._reuse):
            x_1 = conv2d(x, 'Conv1', 32, 3, 2, 'SAME', True, tf.nn.relu,self._is_train )
            x_2 = tf.nn.avg_pool(x_1, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME', name='pooling1')
            x_3 = conv2d(x_2, 'Conv2', 64, 3, 2, 'SAME', True, tf.nn.relu, self._is_train)
            x_4 = tf.nn.avg_pool(x_3, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME', name='pooling2')
            x_5 = conv2d(x_4, 'Conv3', 128, 3, 2, 'SAME', True, tf.nn.relu, self._is_train)
            x_6 = conv2d_transpose(x_5, 64, 3, 2, 'SAME', True, tf.nn.relu, self._is_train)
            x_7 = conv2d(x_6, 'Conv4', 64, 3, 2, 'SAME', True, tf.nn.relu, self._is_train)
            x_8 = conv2d_transpose(x_7, 32, 3, 2, 'SAME', True, tf.nn.relu, self._is_train)

            # x_9,x_10,x_11 as reg layer
            x_9 = conv2d(x_8, 'Reg1', 2, 3, 1, 'SAME', True, tf.nn.relu, self._is_train)
            x_10 = conv2d(x_7, 'Reg2', 2, 3, 1, 'SAME', True, tf.nn.relu, self._is_train)
            x_11 = conv2d(x_5, 'Reg2', 2, 3, 1, 'SAME', True, tf.nn.relu, self._is_train)
        if self._reuse is None:
            self.var_list = tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._name)
            self.saver = tf.train.Saver(self.var_list)
            self._reuse = True
        return x_9, x_10, x_11

    def save(self, session, checkpoint_path):
        self.saver.save(session, checkpoint_path)

    def restore(self, session, checkpoint_path):
        self.saver.restore(session, checkpoint_path)


class fcnRegressor(object):
    def __init__(self, sess: tf.Session, is_train: bool, config: dict):
        # get train parameters
        self._sess = sess
        _is_train = is_train
        _batch_size = config['batch_size']
        _img_height, _img_width = config["image_size"]
        self.x = tf.placeholder(dtype=tf.float32, shape=[_batch_size, _img_height, _img_width, 1])
        self.y = tf.placeholder(dtype=tf.float32, shape=[_batch_size, _img_height, _img_width, 1])