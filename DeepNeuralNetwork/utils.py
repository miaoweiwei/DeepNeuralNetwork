#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019/8/16 16:08
@Author  : miaoweiwei
@File    : utils.py
@Software: PyCharm
@Desc    : 
"""
import os
import cv2
import numpy as np
from vgg16.vgg_preprocess import preprocess_for_train

# 这里使用的是 VggNet 图片的大小是 224X224
img_width = 224
img_height = 224


def img_resize():
    dir = './dogs-vs-cats/train'
    for root, dirs, files in os.walk(dir):
        for file in files:
            filepath = os.path.join(root, file)
            try:
                image = cv2.imread(filepath)
                dim = (224, 224)
                resized = cv2.resize(image, dim)
                if file.split('.')[0] == 'cat':
                    path = './cat_and_dog/train/cat/' + file
                else:
                    path = './cat_and_dog/train/dog/' + file
                a = cv2.imwrite(path, resized)
            except:
                print(filepath)
                os.remove(filepath)
        cv2.waitKey(0)


# 数据输入
def get_file(file_dir):
    images = []
    temp = []
    for root, sub_folders, files in os.walk(file_dir):
        for name in files:
            images.append(os.path.join(root, name))
        for name in sub_folders:  # 把子文件夹存到 temp 中
            temp.append(os.path.join(root, name))
    labels = []
    for one_folder in temp:  # temp 中有cat和dog文件夹的路径
        n_img = len(os.listdir(one_folder))  # 获取one_folder这个文件夹里的文件的个数
        letter = one_folder.split('\\')[-1]  # 获取当前是哪个文件夹
        if letter == 'cat':
            labels = np.append(labels, n_img * [0])  # cat的标签为0
        else:
            labels = np.append(labels, n_img * [1])  # dog的标签为1

    # shuffle
    temp = np.array([images, labels])
    temp = temp.transpose()  # 转置
    np.random.shuffle(temp)  # 按行打乱顺序
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(float(i)) for i in label_list]

    return image_list, label_list


def get_data_from_file(file_dir):
    images = []
    labels = []
    for file in os.listdir(file_dir):
        file_path = os.path.join(file_dir, file)
        images.append(file_path)
        file_name = file.split('.')[0]
        if file_name == 'cat':
            labels.append(0)
        else:
            labels.append(1)

    temp = np.array([images, labels])
    temp = temp.transpose()  # 转置
    np.random.shuffle(temp)  # 按行打乱顺序
    images = list(temp[:, 0])
    labels = list(temp[:, 1])
    labels = [int(float(i)) for i in labels]

    return images, labels


# capacity 表示内存存储的最大容量
def get_batch(image_list, label_list, img_width, img_height, batch_size, capacity):  # 通过读取列表来载入批量图片及标签
    """
    image_list:图片文件路径列表
    label_list:标签列表
    img_width:图像的宽
    img_height:图像的高
    batch_size:一个批次的大小
    capacity:内存中的容量
    """
    import tensorflow as tf
    image = tf.cast(image_list, tf.string)
    label = tf.cast(label_list, tf.int32)

    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])

    image = tf.image.decode_jpeg(image_contents, channels=3)
    # image = preprocess_for_train(image, 224, 224)  # 图像预处理 保持与 vggNet 在 ImageNet上 的预处理形式保持一致
    # image = preprocess_for_train(image, img_width,img_height)  # 图像预处理 保持与 vggNet 在 ImageNet上 的预处理形式保持一致 ，img_width, img_height为处理过后图片的大小

    # 使用下面这两句代码也是可以的
    image = tf.image.resize_image_with_crop_or_pad(image, img_width, img_height)
    image = tf.image.per_image_standardization(image)  # 将图片标准化

    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=64, capacity=capacity)
    label_batch = tf.reshape(label_batch, [batch_size])

    return image_batch, label_batch


# 标签格式的重构
def onehot(labels):
    """进行独热编码"""
    n_sample = len(labels)
    n_class = max(labels) + 1
    onehot_labels = np.zeros((n_sample, n_class))
    onehot_labels[np.arange(n_sample), labels] = 1
    return onehot_labels


if __name__ == '__main__':
    img_resize()