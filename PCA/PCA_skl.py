#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2021-08-26 17:12:43
LastEditor: JiangJi
LastEditTime: 2021-08-26 17:37:19
Discription: 
Environment: 
'''
import sys,os

curr_path = os.path.dirname(os.path.abspath(__file__)) # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path) # 父路径
sys.path.append(parent_path) # 添加路径到系统路径

from Mnist.load_data import load_local_mnist

from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

(X_train, y_train), (X_test, y_test) = load_local_mnist(one_hot=False)
pca = PCA(.95)
lower_dimensional_data = pca.fit_transform(X_train)
print(pca.n_components_)