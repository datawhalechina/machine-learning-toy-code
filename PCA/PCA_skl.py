#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2021-08-26 17:12:43
LastEditor: JiangJi
LastEditTime: 2021-08-27 12:04:21
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

(X_train, y_train), (X_test, y_test) = load_local_mnist(normalize = False,one_hot = False)
print(f"原本特征维度数：{X_train.shape[1]}") # 特征维度数为784

# n_components是>=1的整数时，表示期望PCA降维后的特征维度数
# n_components是[0,1]的数时，表示主成分的方差和所占的最小比例阈值，PCA类自己去根据样本特征方差来决定降维到的维度
pca = PCA(n_components=0.95) 
lower_dimensional_data = pca.fit_transform(X_train)
print(f"降维后的特征维度数：{pca.n_components_}")

approximation = pca.inverse_transform(lower_dimensional_data)

plt.figure(figsize=(8,4));

# 原始图片
plt.subplot(1, 2, 1);
plt.imshow(X_train[1].reshape(28,28),
              cmap = plt.cm.gray, interpolation='nearest',
              clim=(0, 255));
plt.xlabel(f'{X_train.shape[1]} components', fontsize = 14)
plt.title('Original Image', fontsize = 20);

# 降维后的图片
plt.subplot(1, 2, 2);
plt.imshow(approximation[1].reshape(28, 28),
              cmap = plt.cm.gray, interpolation='nearest',
              clim=(0, 255));
plt.xlabel(f'{pca.n_components_} components', fontsize = 14)
plt.title('95% of Explained Variance', fontsize = 20);
plt.show()