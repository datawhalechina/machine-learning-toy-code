#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2021-08-26 11:23:51
LastEditor: JiangJi
LastEditTime: 2021-12-17 15:18:14
Discription: 
Environment: 
'''
import sys,os
curr_path = os.path.dirname(os.path.abspath(__file__)) # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path) # 父路径
p_parent_path = os.path.dirname(parent_path)
sys.path.append(p_parent_path) 
print(f"主目录为：{p_parent_path}")


import time
import numpy as np
import math
import random
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def load_local_mnist():
    ''' 使用Torch加载本地Mnist数据集
    '''
    train_dataset = datasets.MNIST(root = p_parent_path+'/datasets/', train = True,transform = transforms.ToTensor(), download = False)
    test_dataset = datasets.MNIST(root = p_parent_path+'/datasets/', train = False, 
                                transform = transforms.ToTensor(), download = False)
    batch_size = len(train_dataset)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    X_train,y_train = next(iter(train_loader))
    X_test,y_test = next(iter(test_loader))
    X_train,y_train = X_train.cpu().numpy(),y_train.cpu().numpy() # tensor转为array形式)
    X_test,y_test = X_test.cpu().numpy(),y_test.cpu().numpy() # tensor转为array形式)
    X_train = X_train.reshape(X_train.shape[0],784)
    X_test = X_test.reshape(X_test.shape[0],784)    
    return X_train,X_test,y_train,y_test
class SVM:
    def __init__(self, X_train, y_train, gamma = 0.001, C = 200, toler = 0.001):
        '''SVM相关参数初始化
        X_train:训练数据集
        y_train: 训练测试集
        gamma: 高斯核中的gamma
        C:软间隔中的惩罚参数
        toler:松弛变量
        注：
            关于这些参数的初始值：参数的初始值大部分没有强要求，请参照书中给的参考，例如C是调和间隔与误分类点的系数，
            在选值时通过经验法依据结果来动态调整。（本程序中的初始值参考于《机器学习实战》中SVM章节，因为书中也
            使用了该数据集，只不过抽取了很少的数据测试。参数在一定程度上有参考性。）
            如果使用的是其他数据集且结果不太好，强烈建议重新通读所有参数所在的公式进行修改。例如在核函数中σ的值
            高度依赖样本特征值范围，特征值范围较大时若不相应增大σ会导致所有计算得到的核函数均为0
        '''
        self.X_train = X_train       #训练数据集
        self.y_train = np.mat(y_train).T   #训练标签集，为了方便后续运算提前做了转置，变为列向量
        self.gamma = gamma                              #高斯核分母中的σ
        self.C = C                                      #惩罚参数
        self.toler = toler                              #松弛变量
        self.k = self.calc_kernel()                      #核函数（初始化时提前计算）
        self.b = 0                                      #SVM中的偏置b
        self.alpha = [0] * self.X_train.shape[0]   # α 长度为训练集数目
        self.E = [0 * self.y_train[i, 0] for i in range(self.y_train.shape[0])]     #SMO运算过程中的Ei
        self.supportVecIndex = []

    def calc_kernel_1(self):
        '''计算高斯核
        '''
        print("开始计算高斯核...")
        m =  self.X_train.shape[0] # 训练集数量 
        kernel = [[0 for _ in range(m)] for _ in range(m)] 
        for i in tqdm(range(m)):
            #得到式7.90中的X
            x_i = np.expand_dims(self.X_train[i, :],axis=0)
            ''' 由于 x_i * x_j 等于 x_j * x_i，一次计算得到的结果可以
                同时放在k[i][j]和k[j][i]中，这样一个矩阵只需要计算一半即可
                所以小循环直接从i开始
            '''
            for j in range(i, m):
                x_j = np.expand_dims(self.X_train[j, :],axis=0)
                kernel[i][j] = np.exp(-self.gamma * (x_i - x_j).dot((x_i - x_j).T)[0][0])
                kernel[j][i] = np.exp(-self.gamma * (x_i - x_j).dot((x_i - x_j).T)[0][0])
        print("完成计算高斯核函！")
        return kernel
        
    def calc_kernel(self):
        ''' 快速计算高斯核：https://www.scutmath.com/fast_kernel_matrix_generation.html
        '''
        print("开始计算高斯核...")
        X1,X2 = self.X_train, self.X_train
        kernel = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        kernel = np.exp(- self.gamma * kernel)
        print("完成计算高斯核！")
        return kernel
    def is_satisfy_KKT(self, i):
        '''
        查看第i个α是否满足KKT条件
        :param i:α的下标
        :return:
            True：满足
            False：不满足
        '''
        gxi = self.calc_gxi(i)
        yi = self.y_train[i]

        #判断依据参照“7.4.2 变量的选择方法”中“1.第1个变量的选择”
        #式7.111到7.113
        #--------------------
        #依据7.111
        if (math.fabs(self.alpha[i]) < self.toler) and (yi * gxi >= 1):
            return True
        #依据7.113
        elif (math.fabs(self.alpha[i] - self.C) < self.toler) and (yi * gxi <= 1):
            return True
        #依据7.112
        elif (self.alpha[i] > -self.toler) and (self.alpha[i] < (self.C + self.toler)) \
                and (math.fabs(yi * gxi - 1) < self.toler):
            return True
        return False

    def calc_gxi(self, i):
        '''
        计算g(xi)
        依据“7.101 两个变量二次规划的求解方法”式7.104
        :param i:x的下标
        :return: g(xi)的值
        '''
        #初始化g(xi)
        gxi = 0
        #因为g(xi)是一个求和式+b的形式，普通做法应该是直接求出求和式中的每一项再相加即可
        #但是读者应该有发现，在“7.2.3 支持向量”开头第一句话有说到“对应于α>0的样本点
        #(xi, yi)的实例xi称为支持向量”。也就是说只有支持向量的α是大于0的，在求和式内的
        #对应的αi*yi*K(xi, xj)不为0，非支持向量的αi*yi*K(xi, xj)必为0，也就不需要参与
        #到计算中。也就是说，在g(xi)内部求和式的运算中，只需要计算α>0的部分，其余部分可
        #忽略。因为支持向量的数量是比较少的，这样可以再很大程度上节约时间
        #从另一角度看，抛掉支持向量的概念，如果α为0，αi*yi*K(xi, xj)本身也必为0，从数学
        #角度上将也可以扔掉不算
        #index获得非零α的下标，并做成列表形式方便后续遍历
        index = [i for i, alpha in enumerate(self.alpha) if alpha != 0]
 
        #遍历每一个非零α，i为非零α的下标
        for j in index:
            #计算g(xi)
            gxi += self.alpha[j] * self.y_train[j] * self.k[j][i]
        #求和结束后再单独加上偏置b
        gxi += self.b

        #返回
        return gxi

    def calc_Ei(self, i):
        '''
        计算Ei
        根据“7.4.1 两个变量二次规划的求解方法”式7.105
        :param i: E的下标
        :return:
        '''
        #计算g(xi)
        gxi = self.calc_gxi(i)
        #Ei = g(xi) - yi,直接将结果作为Ei返回
        return gxi - self.y_train[i]

    def get_alpha2(self, E1, i):
        '''SMO算法选择第二个变量alpha_2，需要选择使得|E1-E2|最大的点
        '''
        E2 = 0
        #初始化|E1-E2|为-1
        maxE1_E2 = -1
        #初始化第二个变量的下标
        maxIndex = -1
        m =  self.X_train.shape[0] # 训练集数量 
        #这一步是一个优化性的算法
        #实际上书上算法中初始时每一个Ei应当都为-yi（因为g(xi)由于初始α为0，必然为0）
        #然后每次按照书中第二步去计算不同的E2来使得|E1-E2|最大，但是时间耗费太长了
        #作者最初是全部按照书中缩写，但是本函数在需要3秒左右，所以进行了一些优化措施
        #--------------------------------------------------
        #在Ei的初始化中，由于所有α为0，所以一开始是设置Ei初始值为-yi。这里修改为与α
        #一致，初始状态所有Ei为0，在运行过程中再逐步更新
        #因此在挑选第二个变量时，只考虑更新过Ei的变量，但是存在问题
        #1.当程序刚开始运行时，所有Ei都是0，那挑谁呢？
        #   当程序检测到并没有Ei为非0时，将会使用随机函数随机挑选一个
        #2.怎么保证能和书中的方法保持一样的有效性呢？
        #   在挑选第一个变量时是有一个大循环的，它能保证遍历到每一个xi，并更新xi的值，
        #在程序运行后期后其实绝大部分Ei都已经更新完毕了。下方优化算法只不过是在程序运行
        #的前半程进行了时间的加速，在程序后期其实与未优化的情况无异
        #------------------------------------------------------
        #获得Ei非0的对应索引组成的列表，列表内容为非0Ei的下标i
        nozeroE = [i for i, Ei in enumerate(self.E) if Ei != 0]
        #对每个非零Ei的下标i进行遍历
        for j in nozeroE:
            #计算E2
            E2_tmp = self.calc_Ei(j)
            #如果|E1-E2|大于目前最大值
            if math.fabs(E1 - E2_tmp) > maxE1_E2:
                #更新最大值
                maxE1_E2 = math.fabs(E1 - E2_tmp)
                #更新最大值E2
                E2 = E2_tmp
                #更新最大值E2的索引j
                maxIndex = j
        #如果列表中没有非0元素了（对应程序最开始运行时的情况）
        if maxIndex == -1:
            maxIndex = i
            while maxIndex == i:
                #获得随机数，如果随机数与第一个变量的下标i一致则重新随机
                maxIndex = int(random.uniform(0, m))
            #获得E2
            E2 = self.calc_Ei(maxIndex)

        #返回第二个变量的E2值以及其索引
        return E2, maxIndex

    def train(self, n_epochs = 100):
        ''' SMO算法训练
        '''
        # n_epochs: 迭代次数，超过设置次数还未收敛则强制停止
        # alpha_change_flag: 单次迭代中有参数改变则增加1
        i_epoch = 0 
        alpha_change_flag = 1 # alpha_change_flag==0时表示上次迭代没有参数改变，如果遍历一遍都没有参数改变，说明达到收敛状态，可以停止了
        m =  self.X_train.shape[0] # 训练集数量 
        while (i_epoch < n_epochs) and (alpha_change_flag > 0):
            i_epoch += 1 # 迭代步数加1
            print(f"{i_epoch}/{n_epochs}")
            alpha_change_flag = 0  # 新的一轮将参数改变标志位重新置0
            #大循环遍历所有样本，用于找SMO中第一个变量
            for i in range(m):
                # alpha_1 需要选择违反KKT条件最严重的样本点，
                if self.is_satisfy_KKT(i) == False: # 此处简化为只要不满足KKT就选择为alpha_1
                    #如果下标为i的α不满足KKT条件，则进行优化
                    #第一个变量α的下标i已经确定，接下来按照“7.4.2 变量的选择方法”第二步
                    #选择变量2。由于变量2的选择中涉及到|E1 - E2|，因此先计算E1
                    E1 = self.calc_Ei(i)
                    #选择第2个变量
                    E2, j = self.get_alpha2(E1, i)

                    #参考“7.4.1两个变量二次规划的求解方法” P126 下半部分
                    #获得两个变量的标签
                    y1 = self.y_train[i]
                    y2 = self.y_train[j]
                    #复制α值作为old值
                    alpha1_old = self.alpha[i]
                    alpha2_old = self.alpha[j]
                    
                    #依据标签是否一致来生成不同的L和H
                    if y1 != y2:
                        L = max(0, alpha2_old - alpha1_old)
                        H = min(self.C, self.C + alpha2_old - alpha1_old)
                    else:
                        L = max(0, alpha2_old + alpha1_old - self.C)
                        H = min(self.C, alpha2_old + alpha1_old)
                    #如果两者相等，说明该变量无法再优化，直接跳到下一次循环
                    if L == H:   continue

                    #计算α的新值
                    #依据“7.4.1两个变量二次规划的求解方法”式7.106更新α2值
                    #先获得几个k值，用来计算事7.106中的分母η
                    k11 = self.k[i][i]
                    k22 = self.k[j][j]
                    k21 = self.k[j][i]
                    k12 = self.k[i][j]
                    
                    #依据式7.106更新α2，该α2还未经剪切
                    alpha2_new = alpha2_old + y2 * (E1 - E2) / (k11 + k22 - 2 * k12)

                    #剪切α2
                    if alpha2_new < L: alpha2_new = L
                    elif alpha2_new > H: alpha2_new = H
                    #更新α1，依据式7.109
                    alpha1_new = alpha1_old + y1 * y2 * (alpha2_old - alpha2_new)

                    #依据“7.4.2 变量的选择方法”第三步式7.115和7.116计算b1和b2
                    b1_new = -1 * E1 - y1 * k11 * (alpha1_new - alpha1_old) \
                            - y2 * k21 * (alpha2_new - alpha2_old) + self.b
                    b2_new = -1 * E2 - y1 * k12 * (alpha1_new - alpha1_old) \
                            - y2 * k22 * (alpha2_new - alpha2_old) + self.b

                    #依据α1和α2的值范围确定新b
                    if (alpha1_new > 0) and (alpha1_new < self.C):
                        b_new = b1_new
                    elif (alpha2_new > 0) and (alpha2_new < self.C):
                        b_new = b2_new
                    else:
                        b_new = (b1_new + b2_new) / 2

                    #将更新后的各类值写入，进行更新
                    self.alpha[i] = alpha1_new
                    self.alpha[j] = alpha2_new
                    self.b = b_new

                    self.E[i] = self.calc_Ei(i)
                    self.E[j] = self.calc_Ei(j)

                    #如果α2的改变量过于小，就认为该参数未改变，不增加parameterChanged值
                    #反之则自增1
                    if math.fabs(alpha2_new - alpha2_old) >= 0.00001:
                        alpha_change_flag += 1

               

        #全部计算结束后，重新遍历一遍α，查找里面的支持向量
        for i in range(m):
            #如果α>0，说明是支持向量
            if self.alpha[i] > 0:
                #将支持向量的索引保存起来
                self.supportVecIndex.append(i)

    def calcSinglKernel(self, x1, x2):
        '''
        单独计算核函数
        :param x1:向量1
        :param x2: 向量2
        :return: 核函数结果
        '''
        #按照“7.3.3 常用核函数”式7.90计算高斯核
        result = (x1 - x2) * (x1 - x2).T
        result = np.exp(-1 * result / (2 * self.gamma ** 2))
        #返回结果
        return np.exp(result)


    def predict(self, x):
        '''
        对样本的标签进行预测
        公式依据“7.3.4 非线性支持向量分类机”中的式7.94
        :param x: 要预测的样本x
        :return: 预测结果
        '''
        result = 0
        for i in self.supportVecIndex:
            #遍历所有支持向量，计算求和式
            #如果是非支持向量，求和子式必为0，没有必须进行计算
            #这也是为什么在SVM最后只有支持向量起作用
            #------------------
            #先单独将核函数计算出来
            tmp = self.calcSinglKernel(self.X_train[i, :], np.mat(x))
            #对每一项子式进行求和，最终计算得到求和项的值
            result += self.alpha[i] * self.y_train[i] * tmp
        #求和项计算结束后加上偏置b
        result += self.b
        #使用sign函数返回预测结果
        return np.sign(result)

    def test(self, X_test, y_test):
        '''
        测试
        :param X_test:测试数据集
        :param y_test: 测试标签集
        :return: 正确率
        '''
        #错误计数值
        errorCnt = 0
        #遍历测试集所有样本
        for i in range(len(X_test)):
            #打印目前进度
            print('test:%d:%d'%(i, len(X_test)))
            #获取预测结果
            result = self.predict(X_test[i])
            #如果预测与标签不一致，错误计数值加一
            if result != y_test[i]:
                errorCnt += 1
        #返回正确率
        return 1 - errorCnt / len(X_test)



if __name__ == '__main__':
    start = time.time()
    X_train,X_test,y_train,y_test = load_local_mnist()  
    # 初始化SVM类
    model = SVM(X_train[:100], y_train[:100], gamma=0.001,C = 200, toler = 0.001)
    # 开始训练
    print('start to train')
    model.train()
    # 开始测试
    print('start to test')
    accuracy = model.test(X_test[:20], y_test[:20])
    print('the accuracy is:%d'%(accuracy * 100), '%')
    # 打印时间
    print('time span:', time.time() - start)
