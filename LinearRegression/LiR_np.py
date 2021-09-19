#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2021-08-27 11:50:51
LastEditor: JiangJi
LastEditTime: 2021-09-01 11:20:37
Discription: 
Environment: 
'''
import numpy as np
import matplotlib.pyplot as plt

def true_fun(X):
    return 1.5*X + 0.2

np.random.seed(0) # 随机种子
n_samples = 30
'''生成随机数据作为训练集'''
X_train = np.sort(np.random.rand(n_samples)) 
y_train = (true_fun(X_train) + np.random.randn(n_samples) * 0.05).reshape(n_samples,1)
data_X = []
for x in X_train:
    data_X.append([1,x])
data_X = np.array((data_X))

m,p = np.shape(data_X) # m, 数据量 p: 特征数
max_iter = 1000 # 迭代数
weights = np.ones((p,1))  
alpha = 0.1 # 学习率
for i in range(0,max_iter):
    error = np.dot(data_X,weights)- y_train
    gradient = data_X.transpose().dot(error)/m
    weights = weights - alpha * gradient
print("输出参数w:",weights[1:][0]) # 输出模型参数w
print("输出参数:b",weights[0]) # 输出参数b

X_test = np.linspace(0, 1, 100)
plt.plot(X_test, X_test*weights[1][0]+weights[0][0], label="Model") 
plt.plot(X_test, true_fun(X_test), label="True function")
plt.scatter(X_train,y_train) # 画出训练集的点
plt.legend(loc="best")
plt.show()