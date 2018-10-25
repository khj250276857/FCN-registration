#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/4 12:44
# @Author  : Hongjian Kang
# @File    : gen_loss_graph_from_log.py

import os
import numpy as np
import matplotlib.pyplot as plt


def gen_loss_graph_from_log():
    workspace = r"C:\Users\khj\Desktop\实验结果（原始）\DN(batch_size=8,weight=200000,learning_rate=0.01,epoch=500)\log"
    train_log = os.path.join(workspace, "train.log")
    valid_log = os.path.join(workspace, "valid.log")

    with open(train_log, 'r') as f:
        train_y_text = f.read()
    with open(valid_log, 'r') as f:
        valid_y_text = f.read()

    train_y_text = [_.strip() for _ in train_y_text.split("\n") if _ != ""]
    valid_y_text = [_.strip() for _ in valid_y_text.split("\n") if _ != ""]
    train_y_list = [float(_.split(',')[2].split('=')[-1]) for _ in train_y_text]
    valid_y_list = [float(_.split(',')[2].split('=')[-1]) for _ in valid_y_text]

    # train_x_list = np.array(range(len(train_y_list))) * 400
    # valid_x_list = np.array(range(len(valid_y_list))) * 400
    train_x_list = np.array(range(len(train_y_list)))
    valid_x_list = np.array(range(len(valid_y_list)))

    # plt.plot(train_x_list, train_y_list, c="red", label='train_loss')
    plt.plot(valid_x_list, valid_y_list, c="red", label="valid_loss")

    plt.xlabel("epoch")
    plt.ylabel("loss(-NCC)")
    # plt.axis([0, 30, -1.8, -1.0])
    plt.legend(bbox_to_anchor=[1, 1])
    plt.show()


if __name__ == '__main__':
    gen_loss_graph_from_log()