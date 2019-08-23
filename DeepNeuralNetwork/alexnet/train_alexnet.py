#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019/8/22 20:05
@Author  : miaoweiwei
@File    : train_alexnet.py
@Software: PyCharm
@Desc    : 利用Tensorflow对预训练的AlexNet网络进行微调
"""
import tensorflow as tf
import numpy as np
import os
from alexnet.alexnet_model import AlexNet
# from datagenerator import ImageDataGenerator
# from datetime import datetime
# from tensorflow.contrib.data import Iterator
import utils

# 模型保存的路径和文件名。
MODEL_SAVE_PATH = "./model"
MODEL_NAME = "alexnet_model.ckpt"

# 训练集图片所在路径
train_dir = 'D:\\Myproject\\Python\\Datasets\\dogs-vs-cats\\dogs-vs-cats\\train'
# 训练图片的尺寸
image_size = 227
# 训练集中图片总数
total_size = 250000

# 学习率
learning_rate = 0.001
# 训练完整数据集迭代轮数
num_epochs = 10
# 数据块大小
batch_size = 128

# 执行Dropout操作所需的概率值
dropout_rate = 0.5
# 类别数目
num_classes = 2
# 需要重新训练的层
train_layers = ['fc8', 'fc7', 'fc6']

# 读取本地图片，制作自己的训练集，返回image_batch，label_batch
# train, train_label = utils.get_files(train_dir)
train, train_label = utils.get_data_from_file(train_dir)
x, y = utils.get_batch(train, train_label, image_size, image_size, batch_size, 2000)

# 用于计算图输入和输出的TF占位符，每次读取一小部分数据作为当前的训练数据来执行反向传播算法
# x =tf.placeholder(tf.float32,[batch_size,227,227,3],name='x-input')
# y =tf.placeholder(tf.float32,[batch_size,num_classes])
keep_prob = tf.placeholder(tf.float32)

# 定义神经网络结构，初始化模型
model = AlexNet(x, keep_prob, num_classes, train_layers)
# 获得神经网络前向传播的输出
score = model.fc8

# 获得想要训练的层的可训练变量列表
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

# 定义损失函数，获得loss
with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=score, labels=y))

# # 定义反向传播算法（优化算法）
# with tf.name_scope("train"):
#     # 获得所有可训练变量的梯度
#     gradients = tf.gradients(loss, var_list)  # 梯度,就是求导数
#     gradients = list(zip(gradients, var_list))
#
#     # 选择优化算法，对可训练变量应用梯度下降算法更新变量
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate)
#     train_op = optimizer.apply_gradients(grads_and_vars=gradients)  # 使loss最小
#     # 上一句类似下面一句的minimize(loss)
#     # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.name_scope('optimizer'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# 使用前向传播的结果计算正确率
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.cast(tf.argmax(score, 1), tf.int32), y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize an saver for store model checkpoints 加载模型
saver = tf.train.Saver()

# 每个epoch中验证集/测试集需要训练迭代的轮数
train_batches_per_epoch = int(np.floor(total_size / batch_size))

with tf.Session() as sess:
    # 变量初始化
    tf.global_variables_initializer().run()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    try:
        for epoch in range(num_epochs):
            # for step in range(train_batches_per_epoch):
            #     # while not coord.should_stop():
            #     if coord.should_stop():
            #         break
            #     sess.run(train_op, feed_dict={keep_prob: 1.})
            #     if step % 50 == 0:
            #         loss_value, accu = sess.run([loss, accuracy], feed_dict={keep_prob: 1.})
            #         print("Afetr %d training step(s),loss on training batch is %g,accuracy is %g." % (
            #             step, loss_value, accu))
            step = 0
            while not coord.should_stop():
                # sess.run(train_op, feed_dict={keep_prob: dropout_rate})
                sess.run(optimizer, feed_dict={keep_prob: dropout_rate})
                if step % 10 == 0:
                    loss_value, accu = sess.run([loss, accuracy], feed_dict={keep_prob: 1.})
                    print("Afetr %d training step(s),loss on training batch is %g,accuracy is %g." % (
                        step, loss_value, accu))
                step += 1

                if step % 100 == 0:
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH, "epoch{:06d}-step{:06d}.ckpt".format(epoch, step)))
        saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))

    except tf.errors.OutOfRangeError:
        print('done!')
    finally:
        coord.request_stop()
    coord.join(threads)
