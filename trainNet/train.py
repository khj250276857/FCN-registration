import os
import tensorflow as tf
from models.fcn import fcnRegressor
from trainNet.config_folder_guard import config_folder_guard
from trainNet.gen_batches import gen_batches


def train():
    config = config_folder_guard({
        # train_parameters
        'image_size': [128, 128],
        'batch_size': 10,
        'learning_rate': 1e-5,
        'epoch_num': 100,
        'save_interval': 1000,
        'shuffle_batch': True,
        # trainNet data folder
        'checkpoint_dir': r'E:\training data\running data\checkpoints',
        'temp_dir': r'E:\training data\running data\validate'
    })

    #定义验证集和训练集
    train_x_dir = r'E:\training data\pet_ct_train\normolized_pt'
    train_y_dir = r'E:\training data\pet_ct_train\resized_ct'
    batch_x, batch_y = gen_batches(train_x_dir, train_y_dir, {
        'batch_size': config['batch_size'],
        'image_size': config['image_size'],
        'shuffle_batch': config['shuffle_batch']
    })
    # valid_x_dir = r''
    # valid_y_dir = r''
    # valid_x, valid_y = gen_batches(valid_x_dir, valid_y_dir, {
    #     'batch_size': config['batch_size'],
    #     'image_size': config['image_size'],
    #     'shuffle_batch': config['shuffle_batch']
    # })
    # valid_iter_num = len(os.listdir(valid_x_dir)) // config['batch_size']
    config['iteration_num'] = len(os.listdir(train_x_dir)) // config["batch_size"]

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
            _train_L.append(_loss_train)
            print('[TRAIN] epoch={:>3d}, iter={:>5d}, loss={:.4f}'.format(epoch + 1, iter_num + 1, _loss_train))
        print('[TRAIN] epoch={:>3d}, loss={:.4f}..................'.format(epoch + 1, sum(_train_L) / len(_train_L)))

    coord.request_stop()
    coord.join(threads)
    sess.close()

if __name__ == '__main__':
    train()

