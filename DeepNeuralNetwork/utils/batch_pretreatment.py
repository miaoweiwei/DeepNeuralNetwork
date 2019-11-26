#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019/9/4 13:13
@Author  : miaoweiwei
@File    : batch_pretreatment.py
@Software: PyCharm
@Desc    : 
"""
import os
import numpy as np
import tensorflow as tf


def get_batch(image_list, label_list, img_width, img_height, batch_size, shuffle=True, epoch_num=None, num_threads=24,
              capacity=256):
    """ 获取批次
    :param image_list: 图片地址列表
    :param label_list: 标签列表
    :param img_width: 指定图片的宽
    :param img_height: 指定图片的高
    :param batch_size: 批次的大小
    :param shuffle: 是否打乱
    :param epoch_num: 迭代次数
    :param num_threads: 线程的个数
    :param capacity: 内存中容量
    :return: 返回批次 image_batch, label_batch
    """
    import tensorflow as tf
    """通过读取列表来载入批量图片及标签"""
    image = tf.cast(image_list, tf.string)
    label = tf.cast(label_list, tf.int32)

    input_queue = tf.train.slice_input_producer([image, label], num_epochs=epoch_num,
                                                shuffle=shuffle)  # 打乱数据让每一次训练的数据顺序都不一样
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3, try_recover_truncated=True)

    # 对图片进行标准化处理 使其大小一样
    image = tf.image.resize_image_with_crop_or_pad(image, img_width, img_height)
    image = tf.image.per_image_standardization(image)  # 将图片标准化

    # num_epochs=None,生成器可以无限次遍历tensor列表
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=num_threads,
                                              capacity=capacity)
    label_batch = tf.reshape(label_batch, [batch_size])
    return image_batch, label_batch


def get_prediction_batch(image_list, img_width, img_height, batch_size, num_threads=1, capacity=256):
    """ 获取用于预测的图片批次
    :param image_list: 图片地址列表
    :param img_width: 指定图片的宽
    :param img_height: 指定图片的高
    :param batch_size: 批次的大小
    :param num_threads: 线程的个数
    :param capacity: 内存中容量
    :return: 返回批次 image_batch, 文件路径
    """
    import tensorflow as tf
    image_list = sorted(image_list, key=lambda item: os.path.split(item)[-1])
    """通过读取列表来载入批量图片及标签"""
    image = tf.cast(image_list, tf.string)

    paths = np.asarray(image_list)

    # num_epochs=None,生成器可以无限次遍历tensor列表
    input_queue = tf.train.slice_input_producer([image, paths], num_epochs=1, shuffle=False)

    image_contents = tf.read_file(input_queue[0])
    image_paths = input_queue[1]
    image = tf.image.decode_jpeg(image_contents, channels=3, try_recover_truncated=True)

    # 对图片进行标准化处理 使其大小一样
    image = tf.image.resize_image_with_crop_or_pad(image, img_width, img_height)
    image = tf.image.per_image_standardization(image)  # 将图片标准化

    image_batch, image_path_batch = tf.train.batch([image, image_paths],
                                                   batch_size=batch_size,
                                                   num_threads=num_threads,
                                                   capacity=capacity)
    return image_batch, image_path_batch


# 标签格式的重构
def onehot(labels, n_class=100):
    """ 进行独热编码
    :param labels: 标签
    :param n_class:识别年龄这个分类要与网络输出层的个数一致 默认100
    :return:
    """
    n_sample = len(labels)
    onehot_labels = np.zeros((n_sample, n_class))
    onehot_labels[np.arange(n_sample), labels] = 1
    return onehot_labels
