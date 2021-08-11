#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-07-09 15:01:26
@LastEditor: John
LastEditTime: 2021-08-11 15:53:15
@Discription: 
@Environment: python 3.7.7
'''
'''
数据集：Mnist
训练集数量：60000(实际使用：2000)
测试集数量：10000（实际使用：200)
运行机器：CPU i7-9750H
依赖：joblib 0.14.1 sklearn 0.23
------------------------------
运行结果：
    正确率：0.96
    运行时长：0.721
'''
# 
# 导入处于不同目录下的Mnist.load_data

import sys,os
curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)  # 添加父目录到系统目录

from Mnist.load_data import load_local_mnist
from sklearn import svm
from sklearn.metrics import classification_report
import joblib 


if __name__ == "__main__":
   
    train = False # 是否训练

    (x_train, y_train), (x_test, y_test) = load_local_mnist(one_hot=False) # one_hot指对标签y进行one hot编码
    # 这里只截取部分数据
    x_train, y_train= x_train[:2000], y_train[:2000] 
    x_test, y_test = x_test[:200],y_test[:200]
    if train:
        model = svm.SVC(C=200,kernel='rbf',gamma=0.01,cache_size=200,probability=False) # 各系数说明：https://blog.csdn.net/qq_16953611/article/details/82414129
        model.fit(x_train, y_train)
        joblib.dump(model, curr_path+"/SVM_sklearn_model.pkl")
    else:
        model=joblib.load(curr_path+"/SVM_sklearn_model.pkl")
    y_predicted = model.predict(x_test)

    print(classification_report(y_test, y_predicted)) # 分类报告