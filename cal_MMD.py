#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/20 14:27
# @Author  : Hongjian Kang
# @File    : cal_MMD.py

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def _mix_rbf_kernel(X, Y, sigmas, wts=None):
    if wts is None:
        wts = [1] * len(sigmas)

    XX = np.matmul(X, np.transpose(X, [1, 0]))  # 后者转置后，矩阵相乘
    XY = np.matmul(X, np.transpose(Y, [1, 0]))
    YY = np.matmul(Y, np.transpose(Y, [1, 0]))

    X_sqnorms = XX.diagonal()    # 返回矩阵的对角线部分
    Y_sqnorms = YY.diagonal()

    r = lambda x: np.expand_dims(x, 0)
    c = lambda x: np.expand_dims(x, 1)

    K_XX, K_XY, K_YY = 0, 0, 0
    for sigma, wt in zip(sigmas, wts):
        gamma = 1 / (2 * sigma**2)
        K_XX += wt * np.exp(-gamma * (-2 * XX + c(X_sqnorms) + r(X_sqnorms)))
        K_XY += wt * np.exp(-gamma * (-2 * XY + c(X_sqnorms) + r(Y_sqnorms)))
        K_YY += wt * np.exp(-gamma * (-2 * YY + c(Y_sqnorms) + r(Y_sqnorms)))

    return K_XX, K_XY, K_YY, sum(wts)


def mix_rbf_mmd2(X, Y, sigmas=(1,), wts=None, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigmas, wts)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)


def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = K_XX.shape[0]
    n = K_YY.shape[0]

    if biased:
        mmd2 = (np.sum(K_XX) / (m * m)
              + np.sum(K_YY) / (n * n)
              - 2 * np.sum(K_XY) / (m * n))
    else:
        if const_diagonal is not False:
            trace_X = m * const_diagonal
            trace_Y = n * const_diagonal
        else:
            trace_X = np.trace(K_XX)
            trace_Y = np.trace(K_YY)

        mmd2 = ((np.sum(K_XX) - trace_X) / (m * (m - 1))
              + (np.sum(K_YY) - trace_Y) / (n * (n - 1))
              - 2 * np.sum(K_XY) / (m * n))

    return mmd2


def main():
    bandwidths = [2.0, 5.0, 10.0, 20.0, 40.0, 80.0]
    # bandwidths = [0.25, 0.5, 1.0, 2.0, 4.0]
    pt_path = r'C:\Users\khj\Desktop\slice data\5_x\32.png'
    ct_path = r'C:\Users\khj\Desktop\slice data\5_y\32.png'
    new_pt_path = r'C:\Users\khj\Desktop\slice data\5_z\32.png'

    X = np.array(Image.open(pt_path), dtype='float32')
    Y = np.array(Image.open(ct_path), dtype='float32')
    Z = np.array(Image.open(new_pt_path), dtype='float32')
    mmd1 = mix_rbf_mmd2(X, Y, sigmas=bandwidths, wts=None, biased=True)
    mmd2 = mix_rbf_mmd2(Z, Y, sigmas=bandwidths, wts=None, biased=True)

    print(mmd1, mmd2)
    plt.figure('x')
    plt.imshow(X, cmap='gray')
    plt.figure('y')
    plt.imshow(Y, cmap='gray')
    plt.figure('z')
    plt.imshow(Z, cmap='gray')
    plt.show()

if __name__ == '__main__':
    main()
