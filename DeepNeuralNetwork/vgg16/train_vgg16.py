#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019/8/16 16:30
@Author  : miaoweiwei
@File    : train_alexnet.py
@Software: PyCharm
@Desc    : 
"""
import tensorflow as tf
import os
from time import time
from vgg16.VGG16_model import Vgg16
import utils

# log信息共有四个等级，按重要性递增为：
# INFO（通知）<WARNING（警告）<ERROR（错误）<FATAL（致命的
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # 输出 INFO + WARNING + ERROR + FATAL
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # 输出 WARNING + ERROR + FATAL
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 输出 ERROR + FATAL
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 输出 FATAL


if __name__ == '__main__':
    startTime = time()
    batch_size = 25
    capacity = 256  # 内存中存储的最大数据容量
    means = [123.68, 116.779, 103.939]  # VGG训练时图像预处理所减均值R（GB三通道） 已经在vgg类中进行了处理

    # xs, ys = utils.get_file('./cat_and_dog/train')  # 获取图像列表和标签列表
    xs, ys = utils.get_data_from_file("D:/Myproject/Python/Datasets/dogs-vs-cats/dogs-vs-cats/train")  # 获取图像列表和标签列表

    image_batch, label_batch = utils.get_batch(xs, ys, 224, 224, batch_size, capacity)  # 通过读取列表来载入批量的图片及标签

    x = tf.placeholder(tf.float32, [None, 224, 224, 3])
    y = tf.placeholder(tf.int32, [None, 2])  # 对 猫 和 狗  两个类别进行判定

    vgg = Vgg16(x)
    fc8_finetuining = vgg.probs

    loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc8_finetuining, labels=y))  # 交叉熵损失函数
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(loss_function)  # 梯度下降优化器

    # pre = tf.nn.softmax(fc8_finetuining)
    correct_pred = tf.equal(tf.arg_max(fc8_finetuining, 1), tf.arg_max(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)
    vgg.load_weights('./vgg16_weights.npz', sess)  # 通过 npz 格式的文件获取VGG的相应权重参数，从而将权值注入即可实现复用

    # saver = tf.train.Saver()
    saver = vgg.saver()

    coord = tf.train.Coordinator()  # 使用协调器Coordinator来管理线程
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    epoch_start_time = time()

    for i in range(2000):
        images, label = sess.run([image_batch, label_batch])
        labels = utils.onehot(label)  # 用 one-hat 形式对标签进行编码

        sess.run(optimizer, feed_dict={x: images, y: labels})

        if (i + 1) % 10 == 0:
            loss, accuracy_record = sess.run([loss_function, accuracy], feed_dict={x: images, y: labels})

            epoch_end_time = time()
            print("the loss is %f " % loss, "the accuracy is %f" % accuracy_record)
            print("Current epoch{0} takes:{1}".format(i + 1, epoch_end_time - epoch_start_time))
            epoch_start_time = epoch_end_time

        if (i + 1) % 100 == 0:
            saver.save(sess, os.path.join("./model/", 'epoch-{:06d}.ckpt'.format(i)))
            print("-" * 20, 'Model:epoch-{:06d}.ckpt saved successfully'.format(i), "-" * 20)

    saver.save(sess, './model/')
    print("Optimization Finished!")

    coord.request_stop()  # 通知其他线程关闭
    coord.join(threads)  # join 操作等待其他线程结束，其他所有线程关闭之后，这一函数才能返回

    duration = time() - startTime
    print("Train Finished takes:", "{:.2f}".format(duration))
