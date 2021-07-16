#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-05-22 10:55:13
@LastEditor: John
@LastEditTime: 2020-06-08 20:42:24
@Discription:
@Environment: python 3.7.7
'''
# 参考https://github.com/Dod-o/Statistical-Learning-Method_Code/blob/master/KNN/KNN.py

'''
数据集：Mnist
训练集数量：60000
测试集数量：10000（实际使用：200）
------------------------------
运行机器：CPU i7-9750H
超参数：k=25
运行结果：
向量距离使用算法——L2欧式距离
    正确率：0.9698
    运行时长：266.36s
'''

import time
import numpy as np
import sys
import os

# 导入处于不同目录下的Mnist.load_data
parent_path=os.path.dirname(os.path.dirname(sys.argv[0])) # 获取上级目录
sys.path.append(parent_path) # 修改sys.path
from Mnist.load_data import load_local_mnist


class KNN:
    def __init__(self, x_train, y_train, x_test, y_test, k):
        '''
        Args:
            x_train [Array]: 训练集数据
            y_train [Array]: 训练集标签
            x_test [Array]: 测试集数据
            y_test [Array]: 测试集标签
            k [int]: k of kNN
        '''
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        # 将输入数据转为矩阵形式，方便运算
        self.x_train_mat, self.x_test_mat = np.mat(
            self.x_train), np.mat(self.x_test)
        self.y_train_mat, self.y_test_mat = np.mat(
            self.y_test).T, np.mat(self.y_test).T
        self.k = k

    def _calc_dist(self, x1, x2):
        '''计算两个样本点向量之间的距离,使用的是欧氏距离
        :param x1:向量1
        :param x2:向量2
        :return:向量之间的欧式距离
        '''
        return np.sqrt(np.sum(np.square(x1 - x2)))

    def _get_k_nearest(self,x):
        '''
        预测样本x的标记。
        获取方式通过找到与样本x最近的topK个点，并查看它们的标签。
        查找里面占某类标签最多的那类标签
        :param trainDataMat:训练集数据集
        :param trainLabelMat:训练集标签集
        :param x:待预测的样本x
        :param topK:选择参考最邻近样本的数目（样本数目的选择关系到正确率，详看3.2.3 K值的选择）
        :return:预测的标记
        '''
        # 初始化距离列表，dist_list[i]表示待预测样本x与训练集中第i个样本的距离
        dist_list=[0]* len(self.x_train_mat)

        # 遍历训练集中所有的样本点，计算与x的距离
        for i in range( len(self.x_train_mat)):
            # 获取训练集中当前样本的向量
            x0 = self.x_train_mat[i]
            # 计算向量x与训练集样本x0的距离
            dist_list[i] = self._calc_dist(x0, x)

        # 对距离列表排序并返回距离最近的k个训练样本的下标
        # ----------------优化点-------------------
        # 由于我们只取topK小的元素索引值，所以其实不需要对整个列表进行排序，而argsort是对整个
        # 列表进行排序的，存在时间上的浪费。字典有现成的方法可以只排序top大或top小，可以自行查阅
        # 对代码进行稍稍修改即可
        # 这里没有对其进行优化主要原因是KNN的时间耗费大头在计算向量与向量之间的距离上，由于向量高维
        # 所以计算时间需要很长，所以如果要提升时间，在这里优化的意义不大。
        k_nearest_index = np.argsort(np.array(dist_list))[:self.k]  # 升序排序
        return k_nearest_index


    def _predict_y(self,k_nearest_index):
        # label_list[1]=3，表示label为1的样本数有3个，由于此处label为0-9，可以初始化长度为10的label_list
        label_list=[0] * 10
        for index in k_nearest_index:
            one_hot_label=self.y_train[index]
            number_label=np.argmax(one_hot_label)
            label_list[number_label] += 1
        # 采用投票法，即样本数最多的label就是预测的label
        y_predict=label_list.index(max(label_list))
        return y_predict

    def test(self,n_test=200):
        '''
        测试正确率
        :param: n_test: 待测试的样本数
        :return: 正确率
        '''
        print('start test')

        # 错误值计数
        error_count = 0
        # 遍历测试集，对每个测试集样本进行测试
        # 由于计算向量与向量之间的时间耗费太大，测试集有6000个样本，所以这里人为改成了
        # 测试200个样本点，若要全跑，更改n_test即可
        for i in range(n_test):
            # print('test %d:%d'%(i, len(trainDataArr)))
            print('test %d:%d' % (i, n_test))
            # 读取测试集当前测试样本的向量
            x = self.x_test_mat[i]
            # 获取距离最近的训练样本序号
            k_nearest_index=self._get_k_nearest(x)
            # 预测输出y
            y=self._predict_y(k_nearest_index)
            # 如果预测label与实际label不符，错误值计数加1
            if y != np.argmax(self.y_test[i]):
                error_count += 1
            print("accuracy=",1 - (error_count /(i+1)))

        # 返回正确率
        return 1 - (error_count / n_test)


if __name__ == "__main__":

    k=25
    start = time.time()
    (x_train, y_train), (x_test, y_test) = load_local_mnist()
    model=KNN( x_train, y_train, x_test, y_test,k)
    accur=model.test()
    end = time.time()
    print("total acc:",accur)
    print('time span:', end - start)