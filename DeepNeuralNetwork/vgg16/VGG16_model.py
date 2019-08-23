#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019/8/16 16:07
@Author  : miaoweiwei
@File    : VGG16_model.py
@Software: PyCharm
@Desc    : 
"""
import tensorflow as tf
import numpy as np


# 修改VGG模型：全连接层的神经元个数；trainable 参数变动
# （1）预训练的VGG是在ImageNet数据集上进行训练的，对1000个类别进行判定
#    若希望利用已训练模型用于其他分类任务，需要修改最后的全连接层
# （2）在镜像Finetuning对模型重新训练时，对于部分不需要训练的层可以通过设置trainable=False来确保其他过程中不会修改权值

class Vgg16(object):
    def __init__(self, imgs):
        self.parameters = []  # 在类的初始化时加入全局列表，将所需共享的参数加载进来
        self.imgs = imgs
        self.convlayers()
        self.fc_layers()
        self.probs = tf.nn.softmax(self.fc8)  # 输出每个属于各个类别的概率值
        # self.probs = self.fc8  # 输出每个属于各个类别的概率值

    def saver(self):
        return tf.train.Saver()

    def maxpool(self, name, input_data):
        """最大池化 窗口为 2X2 步长为 2 """
        out = tf.nn.max_pool(input_data, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', name=name)
        return out

    def conv(self, name, input_data, out_channel, trainable=False):  # trainablec参数变动
        """卷积层"""
        in_channel = input_data.get_shape()[-1]  # 表示上一层的输出通道数 作为这一层的输入通道数
        with tf.variable_scope(name):
            kernel = tf.get_variable("weights", [3, 3, in_channel, out_channel], dtype=tf.float32, trainable=trainable)
            biases = tf.get_variable("biases", [out_channel], dtype=tf.float32, trainable=trainable)
            conv_res = tf.nn.conv2d(input_data, kernel, [1, 1, 1, 1], padding="SAME")
            res = tf.nn.bias_add(conv_res, biases)
            out = tf.nn.relu(res, name=name)
        self.parameters += [kernel, biases]  # 将卷积层定义的参数（kernel，biases）加入列表
        return out

    def fc(self, name, input_data, out_channel, trainable=True):  # trainablec参数变动
        """全连接层"""
        # 上一层的形状 上一层为卷积层形状为[batch, width, height, output],上一层为全连接层形状就是 [input,output]
        shape = input_data.get_shape().as_list()
        # size 就是全连接层的神经元的个数
        if len(shape) == 4:
            size = shape[-1] * shape[-2] * shape[-3]  # 全连接层输入神经元个数，out_channel 是输出层的 神经元个数
        else:
            size = shape[1]
        input_data_flat = tf.reshape(input_data, [-1, size])  # 对数据进行展开成一维
        with tf.variable_scope(name):
            weights = tf.get_variable(name="weights", shape=[size, out_channel], dtype=tf.float32, trainable=trainable)
            biases = tf.get_variable(name="biases", shape=[out_channel], dtype=tf.float32, trainable=trainable)
            res = tf.matmul(input_data_flat, weights)
            out = tf.nn.relu(tf.nn.bias_add(res, biases))
        self.parameters += [weights, biases]  # 将全连接层定义的参数（weights,biases）加入列表
        return out

    def convlayers(self):
        # zero-mean input
        # conv1
        self.conv1_1 = self.conv("conv1re_1", self.imgs, 64, trainable=False)  # 图片 224*224 输入 3 通道 输出 64 通道
        self.conv1_2 = self.conv("conv1_2", self.conv1_1, 64, trainable=False)
        self.pool1 = self.maxpool("poolre1", self.conv1_2)  # 图片 112*112 输出 64 通道

        # conv2
        self.conv2_1 = self.conv("conv2_1", self.pool1, 128, trainable=False)  # 图片 112*112 输入 64 通道 输出 128 通道
        self.conv2_2 = self.conv("convwe2_2", self.conv2_1, 128, trainable=False)
        self.pool2 = self.maxpool("pool2", self.conv2_2)  # 图片 56*56 输出 128 通道

        # conv3
        self.conv3_1 = self.conv("conv3_1", self.pool2, 256, trainable=False)  # 图片 56*56 输入 128 通道 输出 256 通道
        self.conv3_2 = self.conv("convrwe3_2", self.conv3_1, 256, trainable=False)
        self.conv3_3 = self.conv("convrew3_3", self.conv3_2, 256, trainable=False)
        self.pool3 = self.maxpool("poolre3", self.conv3_3)  # 图片 28*28 输出 256 通道

        # conv4
        self.conv4_1 = self.conv("conv4_1", self.pool3, 512, trainable=False)  # 图片 28*28 输入 256 通道 输出 512 通道
        self.conv4_2 = self.conv("convrwe4_2", self.conv4_1, 512, trainable=False)
        self.conv4_3 = self.conv("conv4rwe_3", self.conv4_2, 512, trainable=False)
        self.pool4 = self.maxpool("pool4", self.conv4_3)  # 图片 14*14 输出 512 通道

        # conv5
        self.conv5_1 = self.conv("conv5_1", self.pool4, 512, trainable=False)  # 图片 14*14 输入 512 通道 输出 512 通道
        self.conv5_2 = self.conv("convrwe5_2", self.conv5_1, 512, trainable=False)
        self.conv5_3 = self.conv("conv5_3", self.conv5_2, 512, trainable=False)
        self.pool5 = self.maxpool("poorwel5", self.conv5_3)  # 图片 7*7 输出 512 通道

    def fc_layers(self, n_class=2):
        self.fc6 = self.fc("fc6", self.pool5, 4096, trainable=False)  # 这一层输入神经元的个数是 7*7*512 输出 4096
        self.fc7 = self.fc("fc7", self.fc6, 4096, trainable=False)  # 这一层输入神经元的个数是 4096 输出 4096
        self.fc8 = self.fc("fc8", self.fc7, 2, trainable=True)  # fc8正是我们需要训练的，因此trainable=True；n_class 是 2,二分类（猫和狗）

    def load_weights(self, weight_file, sess):
        """这个函数将获取的权重载入VGG模型中"""
        print("Begin Load Weight...")
        weights = np.load(weight_file)
        keys = sorted(weights.keys())  # 先对键值对进行排序
        for i, k in enumerate(keys):
            # 30，31 为 fc8层的 weight 和 biases
            if i not in [30, 31]:  # 剔除不需要载入的层 parameters 共有 32 个元素 最后两层需要训练的不用载入
                sess.run(self.parameters[i].assign(weights[k]))
        print("-" * 20, "weights loaded", "-" * 20)
