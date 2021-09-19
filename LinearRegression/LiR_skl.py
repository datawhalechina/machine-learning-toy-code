#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2021-09-01 11:33:06
LastEditor: JiangJi
LastEditTime: 2021-09-01 11:34:08
Discription: 
Environment: 
'''
import numpy as np
from sklearn.linear_model import LinearRegression # 导入线性回归模型
import matplotlib.pyplot as plt

def true_fun(X_train):
    return 1.5*X_train + 0.2

np.random.seed(0) # 随机种子
n_samples = 30
'''生成随机数据作为训练集'''
X_train = np.sort(np.random.rand(n_samples)) 
y_train = (true_fun(X_train) + np.random.randn(n_samples) * 0.05).reshape(n_samples,1)

model = LinearRegression() # 定义模型
model.fit(X_train[:,np.newaxis], y_train) # 训练模型

print("输出参数w:",model.coef_) # 输出模型参数w
print("输出参数:b",model.intercept_) # 输出参数b

X_test = np.linspace(0, 1, 100)
plt.plot(X_test, model.predict(X_test[:, np.newaxis]), label="Model")
plt.plot(X_test, true_fun(X_test), label="True function")
plt.scatter(X_train,y_train) # 画出训练集的点
plt.legend(loc="best")
plt.show()


from sklearn.linear_model import LinearRegression

X_train = [[1,1,1],[1,1,2],[1,2,1]]
y_train = [[6],[9],[8]]
 
model = LinearRegression()
model.fit(X_train, y_train)
print("输出参数w:",model.coef_) # 输出参数w1,w2,w3
print("输出参数b:",model.intercept_) # 输出参数b
test_X = [[1,3,5]]
pred_y = model.predict(test_X)
print("预测结果:",pred_y)