#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/6 21:06
# @Author  : Hongjian Kang
# @File    : logger.py

import os
import logging
import sys


def my_logger(folder_name: str, file_name: str):
    # get logger
    _logger = logging.getLogger(os.path.join(folder_name, file_name))
    _logger.setLevel(logging.INFO)
    # formatter
    formatter = logging.Formatter('%(message)s')
    # console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    # file handler
    if not os.path.exists(folder_name):
        print("[DEBUG] Folder \"{}\" does not exist. create".format(folder_name), file=sys.stdout)
        os.makedirs(folder_name)
    file_handler = logging.FileHandler(os.path.join(folder_name, file_name))
    file_handler.setFormatter(formatter)
    # add the handler to the root logger
    _logger.addHandler(console)
    _logger.addHandler(file_handler)
    return _logger