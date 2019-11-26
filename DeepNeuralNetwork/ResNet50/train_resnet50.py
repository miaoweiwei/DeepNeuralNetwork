#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019/8/27 23:31
@Author  : miaoweiwei
@File    : train_resnet50.py
@Software: PyCharm
@Desc    : 
"""
import os
import math

from utils import dataset, batch_pretreatment
import random
import numpy as np
import pandas as pd
from time import time
import tensorflow as tf
from ResNet50.resnet50_model import ResNet50

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def pretreatment(train_dir, train_csv_path, val_csv_path):
    print("begin pretreatment ... ")
    class_labels, image_list, label_list = dataset.get_flower_file(train_dir)
    # 划分 数据集 默认test_proportion=0.2 分别保存到 train.csv 和 test.csv 文件中
    dataset.division_dataset(image_list, label_list, train_csv=train_csv_path,
                             test_csv=val_csv_path, test_proportion=0.2)
    print("pretreatment complete!")


def train(train_csv_path, val_csv_path, model_dir):
    startTime = time()
    n_class = 102
    batch_size = 12
    capacity = 24  # 内存中存储的最大数据容量

    learning_rate_base = 1e-3  # 最初学习率
    learning_rate_decay = 0.5  # 学习率的衰减率
    learning_rate_step = 1  # 喂入多少轮BATCH-SIZE以后，更新一次学习率。一般为总样本数量/BATCH_SIZE

    xs, ys = dataset.load_dataset(train_csv_path)
    xs_val, ys_val = dataset.load_dataset(val_csv_path)

    images_num = len(xs)  # 总的图片数量
    labels_num = len(ys)  # 总的图片数量
    batch_num = math.ceil(images_num / batch_size)  # 每一个轮次执行的批次数量

    images_val_num = len(xs_val)
    labels_val_num = len(ys_val)
    batch_val_num = math.floor(len(xs_val) / batch_size)  # 每一个轮次执行的批次数量

    print("images:{0} labels:{1} class:{2} batch_size:{3} batch_num:{4}".format(images_num,
                                                                                labels_num,
                                                                                n_class,
                                                                                batch_size,
                                                                                batch_num))
    print("images_val:{0} labels_val:{1} class:{2} batch_size:{3} batch_val_num:{4}".format(images_val_num,
                                                                                            labels_val_num,
                                                                                            n_class,
                                                                                            batch_size,
                                                                                            batch_val_num))

    # 通过读取列表来载入批量的图片及标签
    image_batch, label_batch = batch_pretreatment.get_batch(xs, ys, 224, 224, batch_size, shuffle=False, num_threads=1,
                                                            capacity=capacity)
    image_batch_test, label_batch_test = batch_pretreatment.get_batch(xs_val, ys_val, 224, 224, batch_size,
                                                                      shuffle=False,
                                                                      capacity=capacity)

    gloabl_steps = tf.Variable(0, trainable=False)  # 计数器，用来记录运行了几轮的BATCH_SIZE，初始为0，设置为不可训练
    learning_rate = tf.train.exponential_decay(learning_rate_base,
                                               gloabl_steps,
                                               learning_rate_step,
                                               learning_rate_decay,
                                               staircase=True)

    Y_hat, model_params = ResNet50(input_shape=[224, 224, 3], classes=n_class)

    # Y_hat = tf.sigmoid(Z)

    X = model_params['input']
    Y_true = tf.placeholder(dtype=tf.float32, shape=[None, n_class])

    Z = model_params['out']['Z']  # Z 没有经过了 softmax 层 A 经过了
    A = model_params['out']['A']  # Z 没有经过了 softmax 层 A 经过了

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z, labels=Y_true))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    correct_prediction = tf.equal(tf.arg_max(A, 1), tf.arg_max(Y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver(max_to_keep=10)

    with tf.Session() as sess:
        try:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            if not os.path.exists(model_dir):
                os.mkdir(model_dir)

            # 如果有检查点文件，读取最新的检查点文件，恢复各种变量
            ckpt = tf.train.latest_checkpoint(model_dir)
            if ckpt is not None:
                print("Loading last checkpoint file ...")
                saver.restore(sess, ckpt)
                print("Checkpoint file loading complete!")
                # 加载所有的参数 从这里就可以直接使用模型进行预测，或者继续训练
            else:
                print("There is no checkpoint file that can be loaded!ss")

            coord = tf.train.Coordinator()  # 使用协调器Coordinator来管理线程
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)

            epoch_start_time = time()
            print("Begin train...")
            epoch = 0
            while epoch < 100:
                epoch += 1
                for batch in range(1, batch_num):
                    images, labels = sess.run([image_batch, label_batch])
                    labels = batch_pretreatment.onehot(labels, n_class=n_class)  # 用 one-hat 形式对标签进行编码

                    if batch % 10 == 0:
                        _, l, acc = sess.run([train_step, loss, accuracy], feed_dict={X: images, Y_true: labels})

                        epoch_end_time = time()
                        print("epoch:{0} batch:{1} loss:{2} accuracy:{3} takes:{4}".format(
                            epoch, batch, l, acc, epoch_end_time - epoch_start_time))
                        epoch_start_time = epoch_end_time
                    else:
                        sess.run(train_step, feed_dict={X: images, Y_true: labels})
                    if batch % (batch_num // 2) == 0:
                        val_loss = []
                        val_acc = []
                        for i in range(batch_val_num):
                            images_test, labels_test = sess.run([image_batch_test, label_batch_test])
                            labels_test = batch_pretreatment.onehot(labels_test, n_class=n_class)  # 用 one-hat 形式对标签进行编码

                            l, acc = sess.run([loss, accuracy], feed_dict={X: images_test, Y_true: labels_test})
                            val_loss.append(l)
                            val_acc.append(acc)

                        print(
                            "val_loss:{0} acc:{1}".format(sum(val_loss) / batch_val_num, sum(val_acc) / batch_val_num))

                        saver.save(sess, os.path.join(model_dir, 'epoch-{0}-batch-{1}.ckpt'.format(epoch, batch)),
                                   global_step=epoch)
                        print("-" * 20, 'Model:epoch-{0}-batch-{1}.ckpt saved successfully'.format(epoch, batch),
                              "-" * 20)

            saver.save(sess, os.path.join(model_dir, 'flower_model.ckpt'))
            print("Optimization Finished!")

            coord.request_stop()  # 通知其他线程关闭
            coord.join(threads)  # join 操作等待其他线程结束，其他所有线程关闭之后，这一函数才能返回

            duration = time() - startTime
            print("Train Finished takes:", "{:.2f}".format(duration))
        except Exception as ex:
            print(ex)
        finally:
            saver.save(sess, os.path.join(model_dir, 'flower_model.ckpt'))
            sess.close()


if __name__ == '__main__':
    dataset_path = r"D:\Myproject\Python\Datasets\FlowerData\102flowers\data"
    train_csv_path = "./data/train.csv"
    val_csv_path = "./data/val.csv"
    model_dir = "./model"
    # pretreatment(dataset_path, train_csv_path, val_csv_path)
    train(train_csv_path, val_csv_path, model_dir)
