#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019/8/20 18:23
@Author  : miaoweiwei
@File    : test_vgg16.py
@Software: PyCharm
@Desc    : 
"""
import os

import cv2
import numpy as np
import tensorflow as tf
from vgg16.VGG16_model import Vgg16

if __name__ == '__main__':
    test_path = "D:/Myproject/Python/Datasets/dogs-vs-cats/dogs-vs-cats/test1/"
    model_path = "./model/epoch-001499.ckpt"
    means = [123.68, 116.779, 103.939]  # VGG训练时图像预处理所减均值R（GB三通道）
    x = tf.placeholder(tf.float32, [None, 224, 224, 3])

    with tf.Session() as sess:
        vgg = Vgg16(x)
        fc8_finetuining = vgg.probs  # 即 softmax(fc8)

        saver = tf.train.Saver()
        print("Model restoring...")

        # saver.restore(sess, ./model/)  # 恢复最后保存的模型
        saver.restore(sess, model_path)  # 恢复指定检查点的模型

        img_num = len(os.listdir(test_path))
        img_index = np.random.randint(0, img_num)
        filepath = os.path.join(test_path, "{0}.jpg".format(img_index))
        # filepath = './dogs-vs-cats/test1/21.jpg'  # 狗的图片
        # filepath = './dogs-vs-cats/test1/92.jpg'  # 猫的图片

        image = cv2.imread(filepath, -1)
        image = cv2.resize(image, dsize=(224, 224))

        # for c in range(3):  # 减去均值
        #     img[:, :, c] -= means[c]
        preb = sess.run(fc8_finetuining, feed_dict={x: [image]})
        max_index = np.argmax(preb)  # 索引值最打的就是分类的结果

        if max_index == 0:
            print("This is a cat with possibility %.6f" % preb[:, 0])
        else:
            print("This is a dog with possibility %.6f" % preb[:, 1])

        cv2.imshow("original image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
