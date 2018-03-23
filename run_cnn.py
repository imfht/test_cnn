#!/usr/bin/env python
# -*- coding: utf-8 -*-
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from keras.models import Sequential

number = "3456789"
srcLetter = "qwertyuipasdfghjkzxcvbnm"
srcUpper = srcLetter.upper()
labeldict = {}
all_char = number + srcLetter + srcUpper
for i, j in zip(range(100), all_char):
    labeldict[j] = i

import numpy as np

print(labeldict)
import keras

num_classes = len(all_char)


def code_to_vector(code='3456'):
    """
    将一个四位的验证码转化成我们的dict的编码
    :param code:
    :return:
    """
    c0 = keras.utils.to_categorical(labeldict[code[0]], num_classes)
    c1 = keras.utils.to_categorical(labeldict[code[1]], num_classes)
    c2 = keras.utils.to_categorical(labeldict[code[2]], num_classes)
    c3 = keras.utils.to_categorical(labeldict[code[3]], num_classes)
    c = np.concatenate((c0, c1, c2, c3), axis=0)
    return c


def value2key(value):
    for i in labeldict:
        if labeldict[i] == value:
            return i


def vector_to_code(vector):
    c0 = np.int(np.argmax(vector[0:num_classes]))
    c1 = np.int(np.argmax(vector[num_classes * 1:num_classes * 2]))
    c2 = np.int(np.argmax(vector[num_classes * 2:num_classes * 3]))
    c3 = np.int(np.argmax(vector[num_classes * 3:num_classes * 4]))

    print('value is ', value2key(c0))
    print('value is ', value2key(c1))
    print('value is ', value2key(c2))
    print('value is ', value2key(c3))


height = 120
width = 320
char_num = 4

from gencode import vieCode

if K.image_data_format() == 'channels_first':
    input_shape = (1, height, width)
else:
    input_shape = (height, width, 1)


def get_batch(batch_size=25):
    X = np.zeros([batch_size, height, width])
    Y = np.zeros([batch_size, len(all_char) * char_num])
    while True:
        for i in range(batch_size):
            img, img_str = vieCode().GetCodeImage()
            X[i] = img.convert('L')
            Y[i] = code_to_vector(img_str)

        if K.image_data_format() == 'channels_first':
            x_train = X.reshape(X.shape[0], 1, 120, 320)
        else:
            x_train = X.reshape(X.shape[0], 120, 320, 1)
        yield x_train, Y


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

model.fit_generator(generator=get_batch(), steps_per_epoch=100, epochs=10)
