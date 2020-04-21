#!/usr/bin/env python
# -*- coding: utf-8 -*-
import keras
import numpy as np
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from keras.models import Sequential

from gencode import vieCode

height = 120
width = 320
char_num = 4

all_char = "3456789" + "qwertyuipasdfghjkzxcvbnm" + "qwertyuipasdfghjkzxcvbnm".upper()
num_classes = len(all_char)  # 每个分为多少个类
labeldict = {}
for i, j in zip(range(0, 100), all_char):
    labeldict[j] = i


def code_to_vector(code='3456'):
    """
    将一个四位的验证码转化成我们的dict的编码
    :param code: 四位数的验证码
    :return: 编码过的向量
    """
    c0 = keras.utils.to_categorical(labeldict[code[0]], num_classes)
    c1 = keras.utils.to_categorical(labeldict[code[1]], num_classes)
    c2 = keras.utils.to_categorical(labeldict[code[2]], num_classes)
    c3 = keras.utils.to_categorical(labeldict[code[3]], num_classes)
    c = np.concatenate((c0, c1, c2, c3), axis=0)
    return c


def value2key(value):
    """
    按值索引，找到 key，效率有点低但是不要紧。
    :param value: key-value 中的 value
    :return: key
    """
    for i in labeldict:
        if labeldict[i] == value:
            return i


def vector_to_code(vector):
    """
    将一个 vector 转化成字符
    vector[num_class*x:num_class*(x+1)]是一个长为 len(labeldict)的一维向量
    表示每个字符的概率，所以我们用np.argmax(vector[0:num_classes])来取得每个字符的概率
    然后使用 value2key 把字符的数字编码转化成对应的字符
    :param vector: 编码过的向量/model pridect 的预测结果
    :return: 四个字的字符
    """
    c0 = np.int(np.argmax(vector[0:num_classes]))
    c1 = np.int(np.argmax(vector[num_classes * 1:num_classes * 2]))
    c2 = np.int(np.argmax(vector[num_classes * 2:num_classes * 3]))
    c3 = np.int(np.argmax(vector[num_classes * 3:num_classes * 4]))

    return ''.join([value2key(c0), value2key(c1), value2key(c2), value2key(c3)])


def get_batch(batch_size=25):
    """
    返回一个用来训练的 batch
    :param batch_size: 每次返回多少个 X_train和 Y_train
    :return: X_train,Y_train
    """

    X = np.zeros([batch_size, height, width])
    Y = np.zeros([batch_size, len(all_char) * char_num])
    while True:
        for i in range(batch_size):
            img, img_str = vieCode().GetCodeImage()
            X[i] = img.convert('L')
            Y[i] = code_to_vector(img_str)
            X[i] = 255 - X[i]
        if K.image_data_format() == 'channels_first':
            x_train = X.reshape(X.shape[0], 1, height, width)
        else:
            x_train = X.reshape(X.shape[0], height, width, 1)
        yield x_train, Y


def get_test_batch(batch_size=1):
    X = np.zeros([batch_size, height, width])
    Y = np.zeros([batch_size, len(all_char) * char_num])
    while True:
        for i in range(batch_size):
            img, img_str = vieCode().GetCodeImage()
            X[i] = img.convert('L')
            Y[i] = code_to_vector(img_str)
            X[i] = 255 - X[i]

        if K.image_data_format() == 'channels_first':
            x_train = X.reshape(X.shape[0], 1, height, width)
        else:
            x_train = X.reshape(X.shape[0], height, width, 1)
        return x_train, img_str


if __name__ == '__main__':
    if K.image_data_format() == 'channels_first':
        input_shape = (1, height, width)
    else:
        input_shape = (height, width, 1)
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(5, 9), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 4)))

    model.add(Conv2D(16, kernel_size=(5, 7), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 3)))

    model.add(Flatten())

    model.add(Dense(num_classes * 4, activation='sigmoid'))

    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    for i in range(64):
        model.fit_generator(generator=get_batch(), steps_per_epoch=200)
        x_test, y_test = get_test_batch()
        predict = model.predict(x_test)
        predict_char = vector_to_code(predict[0])
        print('real_char:%s,cnn_get:%s' % (y_test, predict_char))
        model.save('trained.model')
