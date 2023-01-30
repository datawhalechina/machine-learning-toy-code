#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-01-30 09:31:34
LastEditor: JiangJi
LastEditTime: 2023-01-30 09:31:35
Discription: 
'''
'''
此脚本提供两种方法，一种为load_local_mnist，即将本地的.gz文件解码为数据，
一种是利用keras在线下载mnist
'''
import numpy as np
from struct import unpack
import gzip
import os


def __read_image(path):
    with gzip.open(path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        img = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    return img


def __read_label(path):
    with gzip.open(path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        lab = np.frombuffer(f.read(), dtype=np.uint8)
        # print(lab[1])
    return lab


def __normalize_image(image):
    '''__normalize_image 将image的像素值(0-255)归一化
    Args:
        image ([type]): [description]
    Returns:
        [type]: [description]
    '''
    img = image.astype(np.float32) / 255.0
    return img


def __one_hot_label(label):
    '''__one_hot_label 将label进行one-hot编码
    Args:
        label ([type]): 输入为0-9，表示数字标签
    Returns:
        [type]: 输出为二进制编码，比如[0,0,1,0,0,0,0,0,0,0]表示数字2
    '''
    lab = np.zeros((label.size, 10))
    for i, row in enumerate(lab):
        row[label[i]] = 1
    return lab


def load_local_mnist(x_train_path=os.path.dirname(__file__)+'/train-images-idx3-ubyte.gz', y_train_path=os.path.dirname(__file__)+'/train-labels-idx1-ubyte.gz', x_test_path=os.path.dirname(__file__)+'/t10k-images-idx3-ubyte.gz', y_test_path=os.path.dirname(__file__)+'/t10k-labels-idx1-ubyte.gz', normalize=True, one_hot=True):
    '''load_mnist 读取.gz格式的MNIST数据集
    Args:
        x_train_path ([type]): [description]
        y_train_path ([type]): [description]
        x_test_path ([type]): [description]
        y_test_path ([type]): [description]
        normalize (bool, optional): [description]. Defaults to True.
        one_hot (bool, optional): one_hot为True的情况下，标签作为one-hot数组返回
                                  one-hot数组是指[0,0,1,0,0,0,0,0,0,0]这样的数组
    Returns:
        [type]: (训练图像, 训练标签), (测试图像, 测试标签)
        训练集数量为60000，每行包含维度为784=28*28的向量
    '''
    image = {
        'train': __read_image(x_train_path),
        'test': __read_image(x_test_path)
    }

    label = {
        'train': __read_label(y_train_path),
        'test': __read_label(y_test_path)
    }

    if normalize:
        for key in ('train', 'test'):
            image[key] = __normalize_image(image[key])

    if one_hot:
        for key in ('train', 'test'):
            label[key] = __one_hot_label(label[key])

    return (image['train'], label['train']), (image['test'], label['test'])



def load_online_data():  # categorical_crossentropy
    from keras.datasets import mnist
    from keras.utils import np_utils
    import numpy as np
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    number = 10000
    x_train, y_train = x_train[0:number], y_train[0:number]
    x_train = x_train.reshape(number, 28 * 28)
    x_test = x_test.reshape(x_test.shape[0], 28 * 28)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    x_test = np.random.normal(x_test)  # 加噪声

    x_train, x_test = x_train / 255, x_test / 255

    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":

    (x_train, y_train), (x_test, y_test) = load_local_mnist()