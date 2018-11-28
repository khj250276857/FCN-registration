#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/19 17:19
# @Author  : Hongjian Kang
# @File    : gen_graph_from log.py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/4 12:44
# @Author  : Hongjian Kang
# @File    : gen_loss_graph_from_log.py

import os
import numpy as np
import matplotlib.pyplot as plt


def gen_loss_graph_from_log():
    # workspace1 = r"C:\Users\khj\Desktop"

    valid_log1 = r'C:\Users\khj\Desktop\valid1.log'
    valid_log2 = r'C:\Users\khj\Desktop\valid2.log'
    valid_log3 = r'C:\Users\khj\Desktop\valid3.log'
    valid_log4 = r'C:\Users\khj\Desktop\valid4.log'
    valid_log5 = r'C:\Users\khj\Desktop\valid5.log'

    with open(valid_log1, 'r') as f1:
        valid_y_text1 = f1.read()
    with open(valid_log2, 'r') as f2:
        valid_y_text2 = f2.read()
    with open(valid_log3, 'r') as f3:
        valid_y_text3 = f3.read()
    with open(valid_log4, 'r') as f4:
        valid_y_text4 = f4.read()
    with open(valid_log5, 'r') as f5:
        valid_y_text5 = f5.read()

    valid_y_text1 = [_.strip() for _ in valid_y_text1.split("\n") if _ != ""]
    valid_y_list1 = [float(_.split(',')[2].split('=')[-1]) for _ in valid_y_text1]
    valid_x_list1 = np.array(range(len(valid_y_list1)))

    valid_y_text2 = [_.strip() for _ in valid_y_text2.split("\n") if _ != ""]
    valid_y_list2 = [float(_.split(',')[2].split('=')[-1]) for _ in valid_y_text2]
    valid_x_list2 = np.array(range(len(valid_y_list2)))

    valid_y_text3 = [_.strip() for _ in valid_y_text3.split("\n") if _ != ""]
    valid_y_list3 = [float(_.split(',')[2].split('=')[-1]) for _ in valid_y_text3]
    valid_x_list3 = np.array(range(len(valid_y_list3)))

    valid_y_text4 = [_.strip() for _ in valid_y_text4.split("\n") if _ != ""]
    valid_y_list4 = [float(_.split(',')[2].split('=')[-1]) for _ in valid_y_text4]
    valid_x_list4 = np.array(range(len(valid_y_list4)))

    valid_y_text5 = [_.strip() for _ in valid_y_text5.split("\n") if _ != ""]
    valid_y_list5 = [float(_.split(',')[2].split('=')[-1]) for _ in valid_y_text5]
    valid_x_list5 = np.array(range(len(valid_y_list5)))

    plt.plot(valid_x_list1, valid_y_list1, c="red", label="alpha=1")
    plt.plot(valid_x_list2, valid_y_list2, c="blue", label="alpha=0.6")
    plt.plot(valid_x_list3, valid_y_list3, c="green", label="alpha=0.4")
    plt.plot(valid_x_list4, valid_y_list4, c="black", label="alpha=0.2")
    plt.plot(valid_x_list5, valid_y_list5, c="goldenrod", label="alpha=0.01")

    plt.legend(loc='lower right')
    plt.xlabel("epoch")
    plt.ylabel("loss(-NCC)")
    # plt.axis([0, 30, -1.8, -1.0])
    plt.legend(bbox_to_anchor=[1, 1])
    plt.show()


if __name__ == '__main__':
    gen_loss_graph_from_log()