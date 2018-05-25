import os
import tensorflow as tf
import numpy as np
from models.fcn import fcnRegressor


def train():
    config = config_folder_guard({
        # train_parameters
        'image_size': [128, 128],
        'batch_size': 10,
        'learning_rate': 1e-5,
        'iteration_num': 10000,
        'save_interval': 2500,
        # train data folder
        'checkpoint_dir': r'',
        'temp_dir': r''
    })










def gen_batches(x_dir: str, y_dir: str, config: dict):
    """
    给定x文件夹和y文件夹，生成batch tensor的函数
    :param x_dir: Moving Image文件夹绝对路径
    :param y_dir: Fixed Image 文件夹绝对路径
    :param config: config["shuffle_batch"]：是否shuffle batch
                    config["batch_size"]：batch大小
                    config["image_size"]：图像的height和width，tuple类型
    :return: Tensor('batch_x', dtype=float32, shape=[batch_size, img_height, img_width, 1])
            Tensor('batch_y', dtype=float32, shape=[batch_size, img_height, img_width, 1])
    """
    x_arr = [os.path.join(x_dir, _) for _ in os.listdir(x_dir)]
    y_arr = [os.path.join(y_dir, _) for _ in os.listdir(y_dir)]
    # 对绝对路径列表进行排序
    x_arr.sort(key=lambda _: int(os.path.split(_)[-1].split(".")[0].split("_")[0]))
    y_arr.sort(key=lambda _: int(os.path.split(_)[-1].split(".")[0].split("_")[0]))
    # 如果参考图像数量和待配准图像数量不同，那么意味着出错了
    assert len(x_arr) == len(y_arr)

    # 构建输入队列 & batch
    input_queue = tf.train.slice_input_producer([x_arr, y_arr], shuffle=config["shuffle_batch"])
    batch_x, batch_y = tf.train.batch(input_queue, batch_size=config["batch_size"])

    # 定义处理tensor的外部python函数
    def _f(input_tensor, batch_size: int, img_height: int, img_width: int, channels: int):
        _ = np.stack([np.array(Image.open(img_name)) for img_name in input_tensor], axis=0) / 255
        return _.astype(np.float32).reshape([batch_size, img_height, img_width, channels])

    # 应用外部python函数处理tensor
    batch_x = tf.py_func(_f, [batch_x, config["batch_size"], *config["image_size"], 1], tf.float32)
    batch_y = tf.py_func(_f, [batch_y, config["batch_size"], *config["image_size"], 1], tf.float32)

    # 返回batch
    return batch_x, batch_y


def config_folder_guard(config: dict):
    if not os.path.exists(config["checkpoint_dir"]):
        os.makedirs(config["checkpoint_dir"])
    if not os.path.exists(config["temp_dir"]):
        os.makedirs(config["temp_dir"])
    return config
