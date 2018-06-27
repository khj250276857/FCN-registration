import os
import numpy as np
import matplotlib.pyplot as plt


def gen_loss_graph_from_log():
    workspace = r"E:\training data\running data\log"
    train_log = os.path.join(workspace, "train.log")
    valid_log = os.path.join(workspace, "valid.log")

    with open(train_log, 'r') as f:
        train_y_text = f.read()
    with open(valid_log, 'r') as f:
        valid_y_text = f.read()

    train_y_text = [_.strip() for _ in train_y_text.split("\n") if _ != ""]
    valid_y_text = [_.strip() for _ in valid_y_text.split("\n") if _ != ""]
    train_y_list = [float(_.split(",")[1].split("=")[-1]) for _ in train_y_text]
    valid_y_list1 = [float(_.split(",")[2].split("=")[-1]) for _ in valid_y_text]
    # todo: remove it
    valid_y_list2 = [float(_.split(",")[3].split("=")[-1]) for _ in valid_y_text]
    valid_y_list3 = [float(_.split(",")[4].split("=")[-1]) for _ in valid_y_text]

    # train_x_list = np.array(range(len(train_y_list))) * 400
    # valid_x_list = np.array(range(len(valid_y_list))) * 400
    train_x_list = np.array(range(len(train_y_list)))
    valid_x_list = np.array(range(len(valid_y_list1)))

    # plt.plot(train_x_list, train_y_list, c="red", label='train_loss')
    plt.plot(valid_x_list, valid_y_list1, c="red", label="valid_loss_1")
    plt.plot(valid_x_list, valid_y_list2, c="blue", label="valid_loss_2")
    plt.plot(valid_x_list, valid_y_list3, c="green", label="valid_loss_3")
    plt.xlabel("epoch")
    plt.ylabel("loss(-NCC)")
    # plt.axis([0, 30, -1.8, -1.0])
    plt.legend(bbox_to_anchor=[1, 1])
    plt.show()


if __name__ == '__main__':
    gen_loss_graph_from_log()