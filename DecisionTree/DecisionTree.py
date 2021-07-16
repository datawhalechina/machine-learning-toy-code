#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-15 09:41:47
@LastEditor: John
@LastEditTime: 2020-06-17 10:42:41
@Discription: 
@Environment: python 3.7.7
'''
import time
import numpy as np
# 导入处于不同目录下的Mnist.load_data
import os 
import sys
parent_path=os.path.dirname(os.path.dirname(sys.argv[0])) # 获取上级目录
sys.path.append(parent_path) # 修改sys.path
from Mnist.load_data import load_local_mnist


class DecisionTree:
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

        #设置epsilon阈值，ID3算法中需要将信息增益与阈值Epsilon比较，若小于则直接处理后返回T
        #该值的大小在设置上并未考虑太多，观察到信息增益前期在运行中为0.3左右，所以设置了0.1
        self.epsilon_threshhold = 0.1
        self.tree={} # 保存生成的树为字典
    
    def majorClass(labelArr):
        '''
        找到当前标签集中占数目最大的标签
        :param labelArr: 标签集
        :return: 最大的标签
        '''
        #建立字典，用于不同类别的标签技术
        classDict = {}
        #遍历所有标签
        for i in range(len(labelArr)):
            #当第一次遇到A标签时，字典内还没有A标签，这时候直接幅值加1是错误的，
            #所以需要判断字典中是否有该键，没有则创建，有就直接自增
            if labelArr[i] in classDict.keys():
                # 若在字典中存在该标签，则直接加1
                classDict[labelArr[i]] += 1
            else:
                #若无该标签，设初值为1，表示出现了1次了
                classDict[labelArr[i]] = 1
        #对字典依据值进行降序排序
        classSort = sorted(classDict.items(), key=lambda x: x[1], reverse=True)
        #返回最大一项的标签，即占数目最多的标签
        return classSort[0][0]

    def train(self):
        '''其实就是创建决策树的过程
        '''
        #打印信息：开始一个子节点创建，打印当前特征向量数目及当前剩余样本数目
        print('start a node', len(self.x_train[0]), len(self.y_train))
        #将标签放入一个字典中，当前样本有多少类，在字典中就会有多少项
        #也相当于去重，多次出现的标签就留一次。举个例子，假如处理结束后字典的长度为1，那说明所有的样本
        #都是同一个标签，那就可以直接返回该标签了，不需要再生成子节点了。
        classDict = {i for i in self.y_train}
        #如果训练数据中所有实例属于同一类Ck，则置T为单节点数，并将Ck作为该节点的类，返回T
        #即若所有样本的标签一致，也就不需要再分化，返回标记作为该节点的值，返回后这就是一个叶子节点
        if len(classDict) == 1:
            #因为所有样本都是一致的，在标签集中随便拿一个标签返回都行，这里用的第0个（因为你并不知道
            #当前标签集的长度是多少，但运行中所有标签只要有长度都会有第0位。
            return self.y_train[0]
        #如果A为空集，则置T为单节点数，并将D中实例数最大的类Ck作为该节点的类，返回T
        #即如果已经没有特征可以用来再分化了，就返回占大多数的类别
        if len(self.x_train[0]) == 0:
            #返回当前标签集中占数目最大的标签
            return majorClass(self.y_train)

        #否则，按式5.10计算A中个特征值的信息增益，选择信息增益最大的特征Ag
        Ag, EpsilonGet = calcBestFeature(self.x_train, self.y_train)

        #如果Ag的信息增益比小于阈值Epsilon，则置T为单节点树，并将D中实例数最大的类Ck
        #作为该节点的类，返回T
        if EpsilonGet < self.x_train:
            return majorClass(self.y_train)

        #否则，对Ag的每一可能值ai，依Ag=ai将D分割为若干非空子集Di，将Di中实例数最大的
        # 类作为标记，构建子节点，由节点及其子节点构成树T，返回T
        treeDict = {Ag:{}}
        #特征值为0时，进入0分支
        #getSubDataArr(self.x_train, self.y_train, Ag, 0)：在当前数据集中切割当前feature，返回新的数据集和标签集
        treeDict[Ag][0] = createTree(getSubDataArr(self.x_train, self.y_train, Ag, 0))
        treeDict[Ag][1] = createTree(getSubDataArr(self.x_train, self.y_train, Ag, 1))

        return treeDict


if __name__ == "__main__":
    #开始时间
    start = time.time()
    (x_train, y_train), (x_test, y_test) = load_local_mnist(one_hot=False)
    model=DecisionTree(x_train, y_train, x_test, y_test)
    