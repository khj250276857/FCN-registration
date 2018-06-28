#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/28 20:40
# @Author  : Hongjian Kang
# @File    : config_folder_guard.py

import os

def config_folder_guard(config: dict):
    if not os.path.exists(config["checkpoint_dir"]):
        os.makedirs(config["checkpoint_dir"])
    if not os.path.exists(config["temp_dir"]):
        os.makedirs(config["temp_dir"])
    if not os.path.exists(config["log_dir"]):
        os.makedirs(config["log_dir"])
    return config