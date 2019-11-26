#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019/9/4 13:24
@Author  : miaoweiwei
@File    : file_process.py
@Software: PyCharm
@Desc    : 
"""
import configparser
import os


def load_config(config_path):
    conf = configparser.ConfigParser()
    conf.read(config_path, encoding='utf-8')
    # # 读取配置文件里所有的Section
    # print(conf.sections())

    model_path = None
    test_path = None
    thread_num = None
    if 'MAIN' not in conf.sections():
        print("配置文件：{0} 中不存在 ['MAIN'] 这个section".format(config_path))
        return model_path, test_path

    if "modelpath" not in conf.options("MAIN"):
        print("配置文件：{0} 的 ['MAIN']这个section中不存在 ModelPath".format(config_path))
    else:
        model_path = [tupl[-1] for tupl in conf.items("MAIN") if "modelpath" in tupl][0]

    if "testpath" not in conf.options("MAIN"):
        print("配置文件：{0} 的 ['MAIN']这个section中不存在 ModelPath".format(config_path))
    else:
        test_path = [tupl[-1] for tupl in conf.items("MAIN") if "testpath" in tupl][0]

    if "threadnum" not in conf.options("MAIN"):
        print("配置文件：{0} 的 ['MAIN']这个section中不存在 ThreadNum".format(config_path))
    else:
        thread_num = [tupl[-1] for tupl in conf.items("MAIN") if "threadnum" in tupl][0]

    # conf.add_section("Test")  # 添加section到配置文件
    # conf.set("Test", "ip", "11.11.1.1")  # Test section新增ip参数和值
    # conf.write(open(config_path, "w"))  # 写完数据要write一下
    # print(conf.items("Test"))  # 打印刚添加的新内容
    return model_path, test_path, thread_num


def rename_file(file_dir):
    for root, sub_folder, files in os.walk(file_dir):
        for i, file in enumerate(files):
            file_name, ext = os.path.splitext(file)
            if ext != '.jpg':
                os.remove(os.path.join(root, file))
            else:
                os.rename(os.path.join(root, file),
                          os.path.join(root, os.path.split(root)[-1] + "_{:04d}{}".format(i, ext)))

    print("Complete!")


if __name__ == '__main__':
    model_path, test_path = load_config('./ResNet50/config.ini')
    file_dir = r'D:\Myproject\Python\Datasets\flower'
    rename_file(file_dir)
