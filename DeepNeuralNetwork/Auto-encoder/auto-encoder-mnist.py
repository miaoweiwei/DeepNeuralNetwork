#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019/11/26 11:51
@Author  : miaoweiwei
@File    : auto-encoder-mnist.py
@Software: PyCharm
@Desc    : 使用全连接层在MNIST数据集上进行 Auto-Encode
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras import losses
from keras.models import Sequential, Model, Input
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPool2D, Flatten, ZeroPadding1D
from keras.optimizers import SGD, RMSprop, Adam
from keras import regularizers
from keras.activations import relu, softmax, sigmoid, tanh
from keras.utils import np_utils


def load_data(data_count=60000):
    (x_train, y_train), (x_test, y_test) = mnist.load_data(r"D:/Myproject/Python/Datasets/MNIST_data/keras/mnist.npz")

    x_train = x_train[0:data_count]
    y_train = y_train[0:data_count]

    # 把图片转成一维的数据
    # x_train = x_train.reshape(data_count, 28 * 28).astype('float32')
    # x_test = x_test.reshape(x_test.shape[0], 28 * 28).astype('float32')

    # 将数据转成float32类型
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # 数据的归一化
    x_train = x_train / 255. - 0.5
    x_test = x_test / 255. - 0.5

    # label 为 0-9 进行onehot编码 在这里用不到
    # y_train = np_utils.to_categorical(y_train, 10)
    # y_test = np_utils.to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test


# kernel_regularizer：施加在权重上的正则项，为keras.regularizer.Regularizer对象
# bias_regularizer：施加在偏置向量上的正则项，为keras.regularizer.Regularizer对象
# activity_regularizer：施加在输出上的正则项，为keras.regularizer.Regularizer对象
def create_encode_model(input_img):
    encoded = Dense(1000, activation=relu)(input_img)
    encoded = Dense(512, activation=relu)(encoded)
    encoded = Dense(256, activation=relu)(encoded)
    encoded = Dense(128, activation=relu)(encoded)
    encoded = Dense(64, activation=relu)(encoded)
    encoded = Dense(32, activation=relu)(encoded)
    encoded = Dense(16, activation=relu)(encoded)
    encoder_output = Dense(2)(encoded)
    return encoder_output


def create_decoder_model(encoder_output):
    decoded = Dense(16, activation=relu)(encoder_output)
    decoded = Dense(32, activation=relu)(decoded)
    decoded = Dense(64, activation=relu)(decoded)
    decoded = Dense(128, activation=relu)(decoded)
    decoded = Dense(256, activation=relu)(decoded)
    decoded = Dense(512, activation=relu)(decoded)
    decoded = Dense(1000, activation=relu)(decoded)
    decode_output = Dense(784, activation=tanh)(decoded)
    return decode_output


def create_autoencoder(input_img, decoder_output):
    # 构建自编码模型
    autoencoder = Model(inputs=input_img, outputs=decoder_output)
    return autoencoder


def training(autoencoder, x_train, epochs=20, batch_size=256):
    autoencoder.compile(optimizer=RMSprop(lr=0.0015, rho=0.95), loss=losses.mse)
    autoencoder.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, shuffle=True)


def show(encoder, x_test, y_test):
    encoded_imgs = encoder.predict(x_test)
    plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_test, s=3)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data()
    x_train = x_train.reshape(x_train.shape[0], 28 * 28)
    x_test = x_test.reshape(x_test.shape[0], 28 * 28)

    np.random.seed(256)

    input_img = Input(shape=(28 * 28,))
    encoder_output = create_encode_model(input_img)
    decoder_output = create_decoder_model(encoder_output)

    # 构建编码模型
    encoder = Model(inputs=input_img, outputs=encoder_output)
    encoder.summary()

    # 构建autoencode
    autoencoder = create_autoencoder(input_img, decoder_output)
    autoencoder.summary()

    training(autoencoder, x_train, epochs=20, batch_size=256)

    show(encoder, x_train, y_train)
