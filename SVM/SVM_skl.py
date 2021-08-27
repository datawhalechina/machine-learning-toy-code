#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-07-09 15:01:26
@LastEditor: John
LastEditTime: 2021-08-27 18:55:47
@Discription: 
@Environment:
'''

import sys,os
curr_path = os.path.dirname(os.path.abspath(__file__)) # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path) # 父路径
sys.path.append(parent_path) # 添加路径到系统路径

from Mnist.load_data import load_local_mnist
from sklearn import svm
from sklearn.metrics import classification_report



if __name__ == "__main__":
   

    (X_train, y_train), (X_test, y_test) = load_local_mnist(normalize=True,one_hot=False)

    # 截取部分数据，否则程序运行可能超时
    X_train, y_train= X_train[:2000], y_train[:2000] 
    X_test, y_test = X_test[:200],y_test[:200]

    '''系数说明
       C: 软间隔的惩罚系数
       kernel: 核函数
       gamma: 核函数系数，只对rbf,poly,sigmod有效。默认为1/n_features
       cache_size: 训练所需要的内存,以MB为单位,默认200M
    '''
    # 构造svm分类器实例
    model_linear = svm.SVC(C=1.0, kernel='linear') # 线性核
    model_poly = svm.SVC(C=1.0, kernel='poly', degree=3) # 多项式核
    model_rbf = svm.SVC(C=100.0, kernel='rbf', gamma=0.5) # 高斯核1
    model_rbf2 = svm.SVC(C=100.0, kernel='rbf', gamma=0.1) # 高斯核2
    models = [model_linear,model_poly,model_rbf,model_rbf2]
    titles = [  'Linear Kernel',
            'Polynomial Kernel with Degree = 3',
            'Gaussian Kernel with gamma = 0.5',
            'Gaussian Kernel with gamma = 0.1']
    for model, i in zip(models, range(len(models))):
        model.fit(X_train, y_train)
        print(f"{titles[i]}'s score: {model.score(X_test,y_test)}")
  
    # print(classification_report(y_test, y_pred)) # 打印报告