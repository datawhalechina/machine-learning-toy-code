#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-06 17:26:50
@LastEditor: John
LastEditTime: 2021-04-08 00:56:08
@Discription: 
@Environment: python 3.7.7
'''
'''
数据集：Mnist
训练集数量：60000
测试集数量：10000
------------------------------
运行机器：CPU i7-9750H
超参数：学习率 lr=0.001 迭代次数 n_iters=10(实际上一次效果就达到了0.99)
运行结果：
    正确率：0.9933
    运行时长：29.93s
'''
import sys,os
curr_path = os.path.dirname(__file__)
parent_path=os.path.dirname(curr_path) 
sys.path.append(parent_path) # add current terminal path to sys.path

import numpy as np
import time
from Mnist.load_data import load_local_mnist

class LogisticRegression:
    def __init__(self, x_train, y_train, x_test, y_test):
        '''
        Args:
            x_train [Array]: 训练集数据
            y_train [Array]: 训练集标签
            x_test [Array]: 测试集数据
            y_test [Array]: 测试集标签
        '''
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        # 将输入数据转为矩阵形式，方便运算
        self.x_train_mat, self.x_test_mat = np.mat(
            self.x_train), np.mat(self.x_test)
        self.y_train_mat, self.y_test_mat = np.mat(
            self.y_test).T, np.mat(self.y_test).T
        # theta表示模型的参数，即w和b
        self.theta=np.mat(np.zeros(len(x_train[0])))
        self.lr=0.001 # 可以设置学习率优化，使用Adam等optimizier
        self.n_iters=10  # 设置迭代次数
    @staticmethod
    def sigmoid(x):
        '''sigmoid function
        '''
        return 1.0/(1+np.exp(-x))
        
    def _predict(self,x_test_mat):
        P=self.sigmoid(np.dot(x_test_mat, self.theta.T))
        if P >= 0.5:
            return 1
        return 0
    def train(self):
        '''训练过程，可参考伪代码
        '''
        for i_iter in range(self.n_iters):
            for n in range(len(self.x_train)):
                result = self.sigmoid(np.dot(self.x_train_mat[n], self.theta.T))
                error = self.y_train[n]- result
                grad = error*self.x_train_mat[n]
                self.theta+= self.lr*grad
            print('LogisticRegression Model(learning_rate={},i_iter={})'.format(
            self.lr, i_iter+1))
    def save(self):
        '''保存模型参数到本地文件
        '''
        np.save(os.path.dirname(sys.argv[0])+"/theta.npy",self.theta)
    def load(self):
        import os 
        import sys
        self.theta=np.load(os.path.dirname(sys.argv[0])+"/theta.npy")
    def test(self):
         # 错误值计数
        error_count = 0
        #对于测试集中每一个测试样本进行验证
        for n in range(len(self.x_test)):
            y_predict=self._predict(self.x_test_mat[n])
            #如果标记与预测不一致，错误值加1
            if self.y_test[n] != y_predict:
                error_count += 1
            print("accuracy=",1 - (error_count /(n+1)))
        #返回准确率
        return 1 - error_count / len(self.x_test)

def normalized_dataset():
    # 加载数据集，one_hot=False意思是输出标签为数字形式，比如3而不是[0,0,0,1,0,0,0,0,0,0]
    (x_train, y_train), (x_test, y_test) = load_local_mnist(one_hot=False)
    # 将w和b结合在一起，因此训练数据增加一维
    ones_col=[[1] for i in range(len(x_train))] # 生成全为1的二维嵌套列表，即[[1],[1],...,[1]]
    x_train_modified=np.append(x_train,ones_col,axis=1)
    ones_col=[[1] for i in range(len(x_test))] # 生成全为1的二维嵌套列表，即[[1],[1],...,[1]]
    x_test_modified=np.append(x_test,ones_col,axis=1)
    # Mnsit有0-9是个标记，由于是二分类任务，所以将标记0的作为1，其余为0
    # 验证过<5为1 >5为0时正确率在90%左右，猜测是因为数多了以后，可能不同数的特征较乱，不能有效地计算出一个合理的超平面
    # 查看了一下之前感知机的结果，以5为分界时正确率81，重新修改为0和其余数时正确率98.91%
    # 看来如果样本标签比较杂的话，对于是否能有效地划分超平面确实存在很大影响
    y_train_modified=np.array([1 if y_train[i]==1 else 0 for i in range(len(y_train))])
    y_test_modified=np.array([1 if y_test[i]==1 else 0 for i in range(len(y_test))])
    return x_train_modified,y_train_modified,x_test_modified,y_test_modified

if __name__ == "__main__":
    start = time.time()   
    x_train_modified,y_train_modified,x_test_modified,y_test_modified = normalized_dataset()
    model=LogisticRegression(x_train_modified,y_train_modified,x_test_modified,y_test_modified)
    model.train()
    model.save()
    model.load()
    accur=model.test()
    end = time.time()
    print("total acc:",accur)
    print('time span:', end - start)


    
