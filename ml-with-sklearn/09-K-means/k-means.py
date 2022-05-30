#coding=utf-8
#Author:haobo
#Date:2022-4-23

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from sklearn import datasets


# 聚类前
X = np.random.rand(100, 2)
plt.scatter(X[:, 0], X[:, 1], marker='o')

# 初始化我们的质心，从原有的数据中选取K个座位质心
def InitCentroids(X, k):
    index = np.random.randint(0,len(X)-1,k)
    return X[index]

#聚类后,假设k=2
kmeans = KMeans(n_clusters=2).fit(X)
label_pred = kmeans.labels_
plt.scatter(X[:, 0], X[:, 1], c=label_pred)
plt.show()