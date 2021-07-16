#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-07-09 15:01:26
@LastEditor: John
@LastEditTime: 2020-07-09 16:26:02
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
import os 
import sys
parent_path=os.path.dirname(os.path.dirname(sys.argv[0])) # 获取上级目录
sys.path.append(parent_path) # 修改sys.path
from Mnist.load_data import load_local_mnist
from sklearn import datasets, svm, metrics
from sklearn.metrics import classification_report
import joblib 
import time




if __name__ == "__main__":
    start = time.time()
    train = False # 是否训练
    (x_train, y_train), (x_test, y_test) = load_local_mnist(one_hot=False)
    x_train, y_train= x_train[:2000], y_train[:2000]
    x_test, y_test = x_test[:200],y_test[:200]
    
    if train:
        classifier = svm.SVC(C=200,kernel='rbf',gamma=0.01,cache_size=200,probability=False) # 各系数说明：https://blog.csdn.net/qq_16953611/article/details/82414129
        classifier.fit(x_train, y_train)
        joblib.dump(classifier, os.path.dirname(sys.argv[0])+"/SVM_sklearn_model.pkl")
    else:
        classifier=joblib.load(os.path.dirname(sys.argv[0])+"/SVM_sklearn_model.pkl")
    y_predicted = classifier.predict(x_test)
    end = time.time()
    print('time span:', end - start)
    print(classification_report(y_test, y_predicted ))