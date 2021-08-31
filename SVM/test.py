#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2021-08-31 20:31:44
LastEditor: JiangJi
LastEditTime: 2021-08-31 20:49:34
Discription: 
Environment: 
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import seaborn as sns
sns.set()

data = np.array([
    [0.1, 0.7],
    [0.3, 0.6],
    [0.4, 0.1],
    [0.5, 0.4],
    [0.8, 0.04],
    [0.42, 0.6],
    [0.9, 0.4],
    [0.6, 0.5],
    [0.7, 0.2],
    [0.7, 0.67],
    [0.27,0.8],
    [0.5, 0.72]
    ])
 
target = [1] * 6 + [0] * 6
 
# x_line = np.linspace(0, 1, 100)
# y_line = 1 - x_line
# plt.scatter(data[:6, 0], data[:6, 1], marker='o', s=100, lw=3)
# plt.scatter(data[6:, 0], data[6:, 1], marker='x', s=100, lw=3)
# plt.plot(x_line, y_line)
 
 
# linear_svc = svm.SVC(kernel='linear', C=C).fit(data, target)
 
x_min, x_max = data[:, 0].min() - 0.2, data[:, 0].max() + 0.2
y_min, y_max = data[:, 1].min() - 0.2, data[:, 1].max() + 0.2
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.002),
                     np.arange(y_min, y_max, 0.002))
 
# title for the plots
titles = ['SVC with linear kernel',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel']

 
for i, gamma in enumerate([1, 5, 15, 35, 45, 55]):
    rbf_svc = svm.SVC(kernel='rbf', gamma=gamma, C = 0.0001).fit(data, target)
 
    # ravel - flatten
    # c_ - vstack
    # #把后面两个压扁之后变成了x1和x2，然后进行判断，得到结果在压缩成一个矩形
    Z = rbf_svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
 
    plt.subplot(4, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.contourf(xx, yy, Z, cmap=plt.cm.ocean, alpha=0.6)
 
    # Plot the training points
    plt.scatter(data[:6, 0], data[:6, 1], marker='o', color='r', s=100, lw=3)
    plt.scatter(data[6:, 0], data[6:, 1], marker='x', color='k', s=100, lw=3)
 
    plt.title('RBF SVM with $\gamma=$' + str(gamma))

linear_svc = svm.SVC(kernel='linear', C=0.001).fit(data, target)
# ravel - flatten
# c_ - vstack
# #把后面两个压扁之后变成了x1和x2，然后进行判断，得到结果在压缩成一个矩形
Z = linear_svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.subplot(4, 2, 7)
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.contourf(xx, yy, Z, cmap=plt.cm.ocean, alpha=0.6)

# Plot the training points
plt.scatter(data[:6, 0], data[:6, 1], marker='o', color='r', s=100, lw=3)
plt.scatter(data[6:, 0], data[6:, 1], marker='x', color='k', s=100, lw=3)

plt.title('Linear SVM')

linear_svc = svm.SVC(kernel='linear', C=0.001).fit(data, target)
# ravel - flatten
# c_ - vstack
# #把后面两个压扁之后变成了x1和x2，然后进行判断，得到结果在压缩成一个矩形
Z = linear_svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.subplot(4, 2, 7)
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.contourf(xx, yy, Z, cmap=plt.cm.ocean, alpha=0.6)

# Plot the training points
plt.scatter(data[:6, 0], data[:6, 1], marker='o', color='r', s=100, lw=3)
plt.scatter(data[6:, 0], data[6:, 1], marker='x', color='k', s=100, lw=3)

plt.title('Linear SVM')

model_poly = svm.SVC(C=0.0001, kernel='poly', degree=10).fit(data, target) # 多项式核

Z = model_poly.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.subplot(4, 2, 8)
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.contourf(xx, yy, Z, cmap=plt.cm.ocean, alpha=0.6)

# Plot the training points
plt.scatter(data[:6, 0], data[:6, 1], marker='o', color='r', s=100, lw=3)
plt.scatter(data[6:, 0], data[6:, 1], marker='x', color='k', s=100, lw=3)

plt.title('Poly SVM')
plt.show()