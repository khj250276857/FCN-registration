import os
import tensorflow as tf
from models.fcn import fcnRegressor
from trainNet.config_folder_guard import config_folder_guard
from trainNet.gen_batches import gen_batches
from trainNet.logger import my_logger as logger


def train():
    config = config_folder_guard({
        # train_parameters
        'image_size': [256, 256],
        'batch_size': 10,
        'learning_rate': 1e-5,
        'epoch_num': 500,
        'save_interval': 1,
        'shuffle_batch': True,
        # trainNet data folder
        'checkpoint_dir': r'E:\training data\running data\checkpoints',
        'temp_dir': r'E:\training data\running data\validate',
        'log_dir': r'E:\training data\running data\log'
    })

    #定义验证集和训练集
    train_x_dir = r'E:\training data\MR Cardiac\pairs_train\all_moving_pairs'
    train_y_dir = r'E:\training data\MR Cardiac\pairs_train\all_fixed_pairs'
    batch_x, batch_y = gen_batches(train_x_dir, train_y_dir, {
        'batch_size': config['batch_size'],
        'image_size': config['image_size'],
        'shuffle_batch': config['shuffle_batch']
    })
    valid_x_dir = r'E:\training data\MR Cardiac\pairs_validate\all_moving_pairs'
    valid_y_dir = r'E:\training data\MR Cardiac\pairs_validate\all_fixed_pairs'
    valid_x, valid_y = gen_batches(valid_x_dir, valid_y_dir, {
        'batch_size': config['batch_size'],
        'image_size': config['image_size'],
        'shuffle_batch': config['shuffle_batch']
    })
    valid_iter_num = len(os.listdir(valid_x_dir)) // config['batch_size']
    config['iteration_num'] = len(os.listdir(train_x_dir)) // config["batch_size"]

    #定义日志记录器
    train_log = logger(config['log_dir'], 'train.log')
    valid_log = logger(config['log_dir'], 'valid.log')

    #构建网络
    sess = tf.Session()
    reg = fcnRegressor(sess, True, config)
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    #开始训练
    print('start training')
    for epoch in range(config['epoch_num']):
        _train_L = []
        for iter_num in range(config['iteration_num']):
            _bx, _by = sess.run([batch_x, batch_y])
            _loss_train = reg.fit(_bx, _by)
            _train_L.append(_loss_train[0])
            print('[TRAIN] epoch={:>3d}, iter={:>5d}, loss={:.4f}, loss_1={:.4f}, loss_2={:.4f}, loss_3={:.4f}'
                  .format(epoch + 1, iter_num + 1, _loss_train[0], _loss_train[1], _loss_train[2], _loss_train[3]))
        # print('[TRAIN] epoch={:>3d}, loss={:.4f}..................'.format(epoch + 1, sum(_train_L) / len(_train_L)))
        train_log.info('[TRAIN] epoch={:>3d}, loss={:.4f}'.format(epoch + 1, sum(_train_L) / len(_train_L)))

        #放入验证集进行验证
        _valid_L = []
        for j in range(valid_iter_num):
            _valid_x, _valid_y = sess.run([valid_x, valid_y])
            _loss_valid = reg.deploy(None, _valid_x, _valid_y)
            _valid_L.append(_loss_valid[0])
        valid_log.info('[VALID] epoch={:>3d}, loss={:.4f}'.format(epoch + 1, sum(_valid_L) / len(_valid_L)))

        if(epoch + 1) % config['save_interval'] == 0:
            _valid_x, _valid_y = sess.run([valid_x, valid_y])
            reg.deploy(config['temp_dir'], _valid_x, _valid_y)
            reg.save(sess, config['checkpoint_dir'])

    coord.request_stop()
    coord.join(threads)
    sess.close()

if __name__ == '__main__':
    train()
