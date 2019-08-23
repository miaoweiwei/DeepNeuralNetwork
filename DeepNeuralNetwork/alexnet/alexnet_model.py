#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019/8/22 20:12
@Author  : miaoweiwei
@File    : alexnet_model.py
@Software: PyCharm
@Desc    : 
"""

# 定义AlexNet神经网络结构模型

import tensorflow as tf
import numpy as np


# 建立模型图
class AlexNet(object):

    # keep_prob:dropout概率,num_classes:数据类别数,skip_layer
    def __init__(self, x, keep_prob, num_classes, skip_layer, weights_path='DEFAULT'):
        """ AlexNet 构造函数
        :param x:输入
        :param keep_prob:dropout概率
        :param num_classes:数据类别数
        :param skip_layer:
        :param weights_path:权重文件的路径
        """
        self.X = x
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob
        self.SKIP_LAYER = skip_layer
        if weights_path == 'DEFAULT':
            self.WEIGHTS_PATH = 'f:\\python程序\\AlexNet_Protect\\bvlc_alexnet.npy'
        else:
            self.WEIGHTS_PATH = weights_path
        self.create()

    def create(self):
        """ 定义网络结构
        """
        # 第一层：卷积层-->最大池化层-->LRN
        # 11*11的卷积核 96通道输出 步长为 4，4
        self.conv1 = conv_layer(self.X, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        # 窗口大小 3*3 步长 2,2
        pool1 = max_pool(self.conv1, 3, 3, 2, 2, padding='VALID', name='pool1')
        norm1 = lrn(pool1, 2, 2e-05, 0.75, name='norml')  # 数据标准化

        # 第二层：卷积层-->最大池化层-->LRN
        # 卷积核 5*5 256 通道输出 步长 1,1 groups 分2层
        self.conv2 = conv_layer(norm1, 5, 5, 256, 1, 1, groups=2, name='conv2')
        pool2 = max_pool(self.conv2, 3, 3, 2, 2, padding='VALID', name='pool2')
        norm2 = lrn(pool2, 2, 2e-05, 0.75, name='norm2')

        # 第三层：卷积层
        # 卷积核 3*3 384通道 步长 1,1
        self.conv3 = conv_layer(norm2, 3, 3, 384, 1, 1, name='conv3')

        # 第四层：卷积层
        # 卷积核 3*3 384 通道 步长 1,1 分2层
        self.conv4 = conv_layer(self.conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')

        # 第五层：卷积层-->最大池化层
        # 卷积核 3*3 256 通道 步长 1,1 分2层
        self.conv5 = conv_layer(self.conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
        pool5 = max_pool(self.conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

        # 第六层：全连接层
        flattened = tf.reshape(pool5, [-1, 6 * 6 * 256])
        fc6 = fc_layer(flattened, 6 * 6 * 256, 4096, name='fc6')
        dropout6 = dropout(fc6, self.KEEP_PROB)

        # 第七层：全连接层
        fc7 = fc_layer(dropout6, 4096, 4096, name='fc7')
        dropout7 = dropout(fc7, self.KEEP_PROB)

        # 第八层：全连接层，不带激活函数
        self.fc8 = fc_layer(dropout7, 4096, self.NUM_CLASSES, relu=False, name='fc8')

    # 加载神经网络预训练参数,将存储于self.WEIGHTS_PATH的预训练参数赋值给那些没有在self.SKIP_LAYER中指定的网络层的参数
    def load_initial_weights(self, session):
        # 下载权重文件
        weights_dict = np.load(self.WEIGHTS_PATH, encoding='bytes').item()

        for op_name in weights_dict:
            if op_name not in self.SKIP_LAYER:
                with tf.variable_scope(op_name, reuse=True):
                    for data in weights_dict[op_name]:
                        # 偏置项
                        if len(data.shape) == 1:
                            var = tf.get_variable('biases', trainable=False)
                            session.run(var.assign(data))
                        # 权重
                        else:
                            var = tf.get_variable('weights', trainable=False)
                            session.run(var.assign(data))


# 定义卷积层，当groups=1时，AlexNet网络不拆分；当groups=2时，AlexNet网络拆分成上下两个部分。
def conv_layer(input, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='SAME', groups=1):
    """ 卷积层的定义
    :param input:输入数据
    :param filter_height:卷积核的高度
    :param filter_width:卷积核的高宽
    :param num_filters:卷积核的数量
    :param stride_y:y方向的部长
    :param stride_x:x方向的部长
    :param name:该层的名字
    :param padding:是否填充
    :param groups:当groups=2时，AlexNet网络拆分成上下两个部分
    :return:
    """
    input_channels = int(input.get_shape()[-1])  # 获得输入图像的通道数

    convovle = lambda i, k: tf.nn.conv2d(i, k, strides=[1, stride_y, stride_x, 1], padding=padding)  # 创建lambda表达式
    with tf.variable_scope(name) as scope:
        # 创建卷积层所需的权重参数和偏置项参数
        weights = tf.get_variable("weights", shape=[filter_height, filter_width, input_channels / groups, num_filters])
        biases = tf.get_variable("biases", shape=[num_filters])
    if groups == 1:
        conv = convovle(input, weights)
    else:  # 当groups不等于1时，拆分输入和权重 # num_or_size_splits：准备切成几份,axis : 准备在第几个维度上进行切割
        input_groups = tf.split(value=input, num_or_size_splits=groups, axis=3)
        weight_groups = tf.split(value=weights, num_or_size_splits=groups, axis=3)
        output = [convovle(i, k) for i, k in zip(input_groups, weight_groups)]
        # 单独计算完后，再次根据深度连接两个网络  axis 表示在哪个维度上拼接，类似于矩阵的拼接
        conv = tf.concat(values=output, axis=3)
    # 加上偏置项
    bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
    # 激活函数
    relu = tf.nn.relu(bias, name=scope.name)
    return relu


# 定义全连接层
def fc_layer(input, input_channel, output_channel, name, relu=True):
    with tf.variable_scope(name) as scope:
        # 创建权重参数和偏置项
        weights = tf.get_variable("weights", shape=[input_channel, output_channel], trainable=True)
        biases = tf.get_variable("biases", [output_channel], trainable=True)
        # 计算 tf.nn.xw_plus_b 类似 tf.matmul(x, w) + b
        act = tf.nn.xw_plus_b(input, weights, biases, name=scope.name)
        if relu:
            return tf.nn.relu(act)
        else:
            return act


# 定义最大池化层
def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1], strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


# 定义局部响应归一化LPN
def lrn(x, radius, alpha, beta, name, bias=1.0):
    # 数据标准化方法 正则法
    return tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha, beta=beta, bias=bias, name=name)


# 定义dropout
def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)
