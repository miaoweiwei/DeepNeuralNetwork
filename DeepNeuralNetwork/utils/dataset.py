#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019/9/4 13:22
@Author  : miaoweiwei
@File    : dataset.py
@Software: PyCharm
@Desc    : 
"""
import os
import shutil
import imghdr
import random
import numpy as np
import pandas as pd
import configparser
from progressbar import ProgressBar

"""数据的预处理"""


def load_shoeprint_data(data_dir):
    """ 加载文件路径
    :param data_dir:数据集的目录
    :return: 返回图片路径的的列表和info标签文件的路径列表
    """
    print("Start loading files...")
    images_path_list = []
    info_path_list = []
    # 加载 图片 和 标签 文件的路径
    for root, sub_folders, files in os.walk(data_dir):
        for name in files:
            if name.split('.')[-1] == 'txt':
                info_path_list.append(os.path.join(root, name))
            else:
                images_path_list.append(os.path.join(root, name))
    print("Images num:{0}".format(len(images_path_list)))

    return images_path_list, info_path_list


def get_shoeprint_file(file_dir):
    """原始数据处理"""
    # 加载文件路径
    images_path_list, info_path_list = load_shoeprint_data(file_dir)

    print("Check image...")
    progress = ProgressBar()
    error_images = [filename for filename in progress(images_path_list) if check_img(filename) is False]
    print("Number of error image:{0}".format(len(error_images)))
    print("Remove the wrong image...")
    for img in error_images:
        print(img)
        images_path_list.remove(img)

    print("Start loading labels...")
    info_content_list = []
    # 读取 标签 文件
    for file in info_path_list:
        with open(file, 'r') as f:
            f_content = f.read().strip()  # 读取全部并去掉首尾空格
            info_content_list.extend(f_content.split('\n'))
    # 把文件名对应的年龄转成字典的形式
    info_dic = {st.split(',')[0]: st.split(',')[1] for st in info_content_list}
    # 获取对应的标签

    labels = [info_dic[os.path.split(file_path)[-1].split('.')[0]] for file_path in images_path_list]
    print("Labels num:{0}".format(len(labels)))

    temp = np.array([images_path_list, labels])
    temp = temp.transpose()  # 转置
    np.random.shuffle(temp)  # 按行打乱顺序
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(float(i)) for i in label_list]
    print("File loading completed")
    return image_list, label_list


def get_flower_file(data_dir):
    print("Start loading files...")
    images_path_list = []
    class_labels = []
    labels = []
    frist = True
    # 加载 图片 和 标签 文件的路径
    for root, sub_folders, files in os.walk(data_dir):
        if frist is False:
            for file in files:
                labels.append(os.path.split(root)[-1])  # 图片文件夹的名字就是标签
                images_path_list.append(os.path.join(root, file))
        if frist:
            for folder in sub_folders:
                class_labels.append(folder)
            frist = False

    temp = np.array([images_path_list, labels])
    temp = temp.transpose()  # 转置
    np.random.shuffle(temp)  # 按行打乱顺序
    image_list = list(temp[:, 0])
    label_list = list(map(float, temp[:, 1]))
    print("File loading completed")

    return class_labels, image_list, label_list


def division_dataset(image_list, label_list, train_csv, test_csv, test_proportion=0.2):
    """ 划分数据集
    :param image_list: 图片地址列表
    :param label_list: 标签列表
    :param train_csv: 划分后的训练集 要保存到 csv文件名字
    :param test_csv: 划分后的测试集 要保存到 csv文件名字
    :param test_proportion: 测试集的比例
    :return:
    """
    print("Start dividing the data set, please wait...")
    test_proportion = 0 if test_proportion <= 0 else test_proportion if test_proportion <= 1 else 1
    random_list = random.sample(range(0, len(image_list)), int(len(image_list) * test_proportion))
    x_test = [image_list[i] for i in random_list]
    y_test = [label_list[i] for i in random_list]
    x_train = [item for i, item in enumerate(image_list) if i not in random_list]
    y_train = [item for i, item in enumerate(label_list) if i not in random_list]
    # 字典中的key值即为csv中列名
    data_train = pd.DataFrame({'image_path': x_train, 'label': y_train})
    data_test = pd.DataFrame({'image_path': x_test, 'label': y_test})
    print("Save csv file")
    # 将DataFrame存储为csv,index表示是否显示行名，default=True
    data_train.to_csv(train_csv, index=False, sep=',', mode='w', encoding='utf-8')
    data_test.to_csv(test_csv, index=False, sep=',', mode='w', encoding='utf-8')
    print("Data set partitioning completed")
    return x_train, y_train, x_test, y_test


def load_dataset(data_csv):
    """ 加载 标注的csv文件
    :param data_csv:
    :return: 返回 图片地址的列表 和 标签的列表
    """
    print("Start loading csv...")
    data = pd.read_csv(data_csv, encoding='utf-8')
    images = data['image_path']
    labels = data['label']
    print("CSV loading completed")
    return list(images), list(labels)


def check_img(image_path):
    if imghdr.what(image_path) is None:  # 这种直接就是打不开图片
        return False
    return True


def check_image(image_path, error_path=None):
    print("Check image...")
    # 加载文件路径
    images_path_list, info_path_list = load_shoeprint_data(image_path)
    original_images = sorted(images_path_list)

    progress = ProgressBar()
    error_images = [filename for filename in progress(original_images) if check_img(filename) is False]
    print("Error number:{0}".format(len(error_images)))
    for error in error_images:
        print(error)
    if error_path is not None:
        error_file = pd.DataFrame({'error': error_images})
        error_file.to_csv(error_path, index=False, sep=',', mode='w', encoding='utf-8')


def class_to_dir():
    data_dir = r'D:\Myproject\Python\Datasets\FlowerData\102flowers\jpg'
    output = r'D:\Myproject\Python\Datasets\FlowerData\102flowers\data'
    images_path_list = []
    for root, sub_folders, files in os.walk(data_dir):
        for filename in files:
            images_path_list.append(os.path.join(root, filename))
    label_csv = r'D:\Myproject\Python\Datasets\FlowerData\102flowers\imageslabels.csv'
    data = pd.read_csv(label_csv, encoding='utf-8')
    labels = list(data['label'])

    progress = ProgressBar()

    for image_path in progress(images_path_list):
        root, file_name = os.path.split(image_path)
        index = int(os.path.splitext(file_name)[0].split('_')[-1]) - 1
        file_dir = os.path.join(output, str(labels[index] - 1))
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)
        shutil.move(image_path, os.path.join(file_dir, file_name))


if __name__ == '__main__':
    # get_age_segment_dic()
    # check_image('/home/guest/Documents/Blind_test/Blind_test', './data/blind_test_error.csv')
    # xs, ys = get_file('/home/guest/Documents/train_data/train_data')
    # onehot_labels = onehot(ys)

    # class_labels, image_list, label_list = get_flower_file(r'D:\Myproject\Python\Datasets\FlowerData\17flowers\jpg')
    # 划分 数据集 默认test_proportion=0.2 分别保存到 train.csv 和 test.csv 文件中
    # division_dataset(image_list, label_list, test_proportion=0.2, train_csv='../data/train.csv',
    #                  test_csv='../data/test.csv')
    # load_dataset("./train.csv")
    class_to_dir()
