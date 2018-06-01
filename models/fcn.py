import tensorflow as tf
from models.utils import conv2d, conv2d_transpose, reg
from models.WarpST import WarpST
from models.utils import ncc, save_image_with_scale
import os

class FCN(object):
    def __init__(self, name: str, is_train: bool):
        self._name = name
        self._is_train = is_train
        self._reuse = None

    def __call__(self, x):
        with tf.variable_scope(self._name, reuse=self._reuse):
            x_1 = conv2d(x, 'Conv1', 32, 3, 2, 'SAME', True, tf.nn.relu, self._is_train)
            x_2 = tf.nn.avg_pool(x_1, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME', name='pooling1')
            x_3 = conv2d(x_2, 'Conv2', 64, 3, 2, 'SAME', True, tf.nn.relu, self._is_train)
            x_4 = tf.nn.avg_pool(x_3, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME', name='pooling2')
            x_5 = conv2d(x_4, 'Conv3', 128, 3, 2, 'SAME', True, tf.nn.relu, self._is_train)
            x_6 = conv2d_transpose(x_5, 'deconv1', 64, [8, 16, 16, 64], 3, 2, 'SAME', True, tf.nn.relu, self._is_train)
            x_7 = conv2d(x_6, 'Conv4', 64, 3, 2, 'SAME', True, tf.nn.relu, self._is_train)
            x_8 = conv2d_transpose(x_7, 'deconv2', 32, [8, 16, 16, 32], 3, 2, 'SAME', True, tf.nn.relu, self._is_train)
            # todo: change batch_size for conv2d_transpose output_shape x_6,x_8
            # x_9,x_10,x_11 as regression layer, corresponding Reg1,Reg2 and Reg3 respectively
            x_9 = reg(x_8, 'Reg1', 2, 3, 1, 'SAME', self._is_train)
            x_10 = reg(x_7, 'Reg2', 2, 3, 1, 'SAME', self._is_train)
            x_11 = reg(x_5, 'Reg3', 2, 3, 1, 'SAME', self._is_train)
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
        _img_height, _img_width = config["image_size"]
        self.x = tf.placeholder(dtype=tf.float32, shape=[_batch_size, _img_height, _img_width, 1])
        self.y = tf.placeholder(dtype=tf.float32, shape=[_batch_size, _img_height, _img_width, 1])
        xy = tf.concat([self.x, self.y], axis=3)    # [batch_size, img_height, img_width, 2]

        # construct Spatial Transformers
        self._fcn = FCN('FCN', is_train=_is_train)
        fcn_out = self._fcn(xy)
        self._z1 = WarpST(self.x, fcn_out[0], [_img_height, _img_width], name="WrapST_1")
        self._z2 = WarpST(self.x, fcn_out[1], [_img_height, _img_width], name="WrapST_2")
        self._z3 = WarpST(self.x, fcn_out[2], [_img_height, _img_width], name="WrapST_3")

        # calculate loss
        self.loss1 = -ncc(self.y, self._z1)
        self.loss2 = -ncc(self.y, self._z2)
        self.loss3 = -ncc(self.y, self._z3)
        self.loss = self.loss1 + 0.6 * self.loss2 + 0.3 * self.loss3

        # construct trainNet step
        if _is_train:
            _optimizer = tf.train.AdadeltaOptimizer(config['learning_rate'])
            _var_list = self._fcn.var_list
            self.train_step = _optimizer.minimize(self.loss, var_list=_var_list)

        # initialize all variables
        self._sess.run(tf.global_variables_initializer())

    def fit(self, batch_x, batch_y):
        _, loss, loss1, loss2, loss3 = self._sess.run(
            fetches=[self.train_step, self.loss, self.loss1, self.loss2, self.loss3],
            feed_dict={self.x: batch_x, self.y: batch_y}
        )
        return loss, loss1, loss2, loss3

    def deploy(self, dir_path, x, y, img_start_idx=0):
        # ca
        z1, z2, z3 = self._sess.run([self._z1, self._z2, self._z3], feed_dict={self.x: x, self.y: y})
        loss, loss_1, loss_2, loss_3 = self._sess.run([self.loss, self.loss1, self.loss2, self.loss3],
                                                      feed_dict={self.x: x, self.y: y})
        if dir_path is None:
            return loss, loss_1, loss_2, loss_3
        else:
            # save image
            for i in range(z1.shape[0]):
                _idx = img_start_idx + i + 1
                save_image_with_scale(dir_path + '/{:>02d}_x.png'.format(_idx), x[i, :, :, 0])
                save_image_with_scale(dir_path + '/{:>02d}_y.png'.format(_idx), y[i, :, :, 0])
                save_image_with_scale(dir_path + '/{:>02d}_z1.png'.format(_idx), z1[i, :, :, 0])
                save_image_with_scale(dir_path + '/{:>02d}_z2.png'.format(_idx), z2[i, :, :, 0])
                save_image_with_scale(dir_path + '/{:>02d}_z2.png'.format(_idx), z3[i, :, :, 0])
            return loss, loss_1, loss_2, loss_3

    def save(self, sess, save_folder: str):
        self._fcn.save(sess, os.path.join(save_folder, 'FCN.ckpt'))

    def restore(self, sess, save_folder: str):
        self._fcn.restore(sess, os.path.join(save_folder, 'FCN.ckpt'))