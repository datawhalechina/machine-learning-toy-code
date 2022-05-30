#coding=utf-8
#Author:haobo
#Date:2022-4-23

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


#加载数据集
digits = load_digits()
data = digits.data     # 特征集
target = digits.target # 目标集


#将数据集拆分为训练集（75%）和测试集（25%）:
train_x, test_x, train_y, test_y = train_test_split(
    data, target, test_size=0.25, random_state=33)


#构造KNN分类器：采用默认参数
knn = KNeighborsClassifier() 


#拟合模型：
knn.fit(train_x, train_y) 
#预测数据：
predict_y = knn.predict(test_x) 


#计算模型准确度
score = accuracy_score(test_y, predict_y)
print(score)