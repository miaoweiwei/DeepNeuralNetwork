#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019/8/29 14:02
@Author  : miaoweiwei
@File    : test_resnet50.py
@Software: PyCharm
@Desc    : 
"""
import os
import math
import utils
from time import time
import tensorflow as tf
from ResNet50.resnet50_model import ResNet50

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def test(test_path_csv, model_dir):
    startTime = time()
    age_segment_dic, age_segment_num = utils.get_age_segment_dic()
    # n_class = age_segment_num
    n_class = 85
    batch_size = 64
    capacity = 256  # 内存中存储的最大数据容量
    xs, ys = utils.load_dataset(test_path_csv)  # 获取数据划分后的训练集，获取图像列表和标签列表，排列顺序已经打乱
    images_num = len(xs)  # 总的图片数量
    labels_num = len(ys)
    batch_num = math.ceil(images_num / batch_size) + 1  # 每一个轮次执行的批次数量

    print("images:{0} labels:{1} class:{2} batch_size:{3} batch_num:{4}".format(images_num,
                                                                                labels_num,
                                                                                n_class,
                                                                                batch_size,
                                                                                batch_num))

    # 通过读取列表来载入批量的图片及标签
    image_batch, label_batch = utils.get_batch(xs, ys, 224, 224, batch_size,
                                               shuffle=False, epoch_num=1, capacity=capacity)

    Y_hat, model_params = ResNet50(input_shape=[224, 224, 3], classes=n_class)

    X = model_params['input']
    Y_true = tf.placeholder(dtype=tf.float32, shape=[None, n_class])

    Z = model_params['out']['Z']  # Z 没有经过了 softmax 层 A 经过了
    A = model_params['out']['A']  # Z 没有经过了 softmax 层 A 经过了

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z, labels=Y_true))

    correct_prediction = tf.equal(tf.arg_max(A, 1), tf.arg_max(Y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        try:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            if not os.path.exists(model_dir):
                return  # 模型文件夹不存在
            # 如果有检查点文件，读取最新的检查点文件，恢复各种变量
            ckpt = tf.train.latest_checkpoint(model_dir)
            print(ckpt)
            if ckpt is not None:
                print("Loading last checkpoint file ...")
                saver.restore(sess, ckpt)
                print("Checkpoint file loading complete!")
                # 加载所有的参数 从这里就可以直接使用模型进行预测，或者继续训练
            else:
                print("There is no checkpoint file that can be loaded!ss")
                return  # 模型不存在

            # saver.restore(sess, "./model/epoch-1-batch-1000.ckpt")

            coord = tf.train.Coordinator()  # 使用协调器Coordinator来管理线程
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)

            def get_age_segment(age):
                for key, value in age_segment_dic.items():
                    if age in value:
                        return key
                else:
                    return -1

            epoch_start_time = time()
            print("Begin test...")
            batch = 0
            try:
                loss_list = []
                acc_list = []
                value0_list = []
                value1_list = []
                value2_list = []
                while not coord.should_stop():
                    batch += 1
                    images, labels = sess.run([image_batch, label_batch])
                    # labels = [get_age_segment(item) for item in labels]  # 年龄映射分类
                    labels = utils.onehot(labels, n_class=n_class)  # 用 one-hat 形式对标签进行编码

                    l, acc, pre, y = sess.run([loss, accuracy, tf.arg_max(A, 1), tf.arg_max(Y_true, 1)],
                                              feed_dict={X: images, Y_true: labels})

                    diff = abs(pre - y)
                    valur0 = sum((diff == 0) + 0)
                    valur1 = sum((diff == 1) + 0)
                    valur2 = sum((diff == 2) + 0)

                    loss_list.append(l)
                    acc_list.append(acc)
                    value0_list.append(valur0)
                    value1_list.append(valur1)
                    value2_list.append(valur2)

                    epoch_end_time = time()
                    # print("batch:{0} loss:{1} accuracy:{2} takes:{3}".format(
                    #     batch, l, acc, epoch_end_time - epoch_start_time))
                    # print("0 nums:{0} 1 nums:{1} 2 nums:{2}".format(valur0, valur1, valur2))

                    if batch % 20 == 0:
                        print("batch:{0} mean loss:{1} mean acc:{2}".format(batch, sum(loss_list) / batch,
                                                                            sum(acc_list) / batch))
                        print("0 total nums:{0} 1 total nums:{1} 2 total nums:{2}".format(sum(value0_list),
                                                                                          sum(value1_list),
                                                                                          sum(value2_list)))

                        print("sc:{0}".format(
                            (sum(value0_list) + sum(value1_list) + sum(value2_list)) / (batch * batch_size)))
                        print("-" * 20, "takes:{0}".format(epoch_end_time - epoch_start_time), "-" * 20)

                    epoch_start_time = epoch_end_time

            except tf.errors.OutOfRangeError:
                print("complete")
            finally:
                coord.request_stop()  # 通知其他线程关闭
                coord.join(threads)  # join 操作等待其他线程结束，其他所有线程关闭之后，这一函数才能返回

                print("mean loss:{0} mean acc:{1}".format(sum(loss_list) / batch, sum(acc_list) / batch))
                print("0 total nums:{0} 1 total nums:{1} 2 total nums:{2}".format(sum(value0_list),
                                                                                  sum(value1_list),
                                                                                  sum(value2_list)))

                print("sc:{0}".format((sum(value0_list) + sum(value1_list) + sum(value2_list)) / (batch * batch_size)))

                duration = time() - startTime
                print("Test Finished takes:", "{:.2f}".format(duration))
        except Exception as ex:
            print(ex)
        finally:
            sess.close()


if __name__ == '__main__':
    test_path_csv = "../data/test_all.csv"
    model_dir = "./model"
    test(test_path_csv, model_dir)
