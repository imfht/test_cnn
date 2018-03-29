#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from optparse import OptionParser

from PIL import Image
from keras.models import load_model

from run_cnn import *


def file2xtrain(file_name, base_dir='./'):
    batch_size = 1
    X = np.zeros([batch_size, height, width])
    while True:
        for i in range(batch_size):
            img = Image.open(file_name)
            X[i] = img.convert('L')
            X[i] = 255 - X[i]
        if K.image_data_format() == 'channels_first':
            x_train = X.reshape(X.shape[0], 1, 120, 320)
        else:
            x_train = X.reshape(X.shape[0], 120, 320, 1)
        return x_train


def main():
    parser = OptionParser()
    parser.add_option('-d', '--dir', help='test all png file in a directory')
    parser.add_option('-r', '--random', help='gen a code and predict with model', action='store_true')
    parser.add_option('-i', '--input', help='test on only one png')
    parser.add_option('-m', '--model', help='the path to model file', default='./trained.model')
    (options, args) = parser.parse_args()
    if not (options.dir or options.input or options.random):
        parser.print_help()
        return
    else:
        model = load_model(options.model)
    if options.dir:
        if not options.dir.endwith('/'):
            options.dir = options.dir + '/'
        for i in os.listdir(options.dir):
            predict = model.predict(file2xtrain(i))
            img_name = vector_to_code(predict[0])
            os.rename(options.dir + i, options.dir + img_name + '.png')
    elif options.input:
        predict = model.predict(file2xtrain(file_name=options.input))
        img_name = vector_to_code(predict[0])
        print 'The input picture seems %s, am I right? ٩꒰ ⑅>∀<⑅ ꒱۶﻿' % img_name
    elif options.random:
        x_test, y_test = get_test_batch()
        predict = model.predict(x_test)
        predict_char = vector_to_code(predict[0])
        if predict_char == y_test:
            print('Predict succeed, code is %s ٩꒰ ⑅>∀<⑅ ꒱۶' % predict_char)
        else:
            print('Sad, real code is : %s , I guess it is: %s' % (y_test, predict_char))
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
