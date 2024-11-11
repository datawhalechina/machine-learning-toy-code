import sys,os
curr_path = os.path.dirname(os.path.abspath(__file__)) # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path) # 父路径
p_parent_path = os.path.dirname(parent_path)
sys.path.append(p_parent_path)
print(f"主目录为：{p_parent_path}")

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.optimize import fmin_cg
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def load_local_mnist():
    ''' 使用Torch加载本地Mnist数据集
    '''
    train_dataset = datasets.MNIST(root = p_parent_path+'/datasets/', train = True,transform = transforms.ToTensor(), download = False)
    test_dataset = datasets.MNIST(root = p_parent_path+'/datasets/', train = False,
                                transform = transforms.ToTensor(), download = False)
    batch_size = len(train_dataset)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    X_train, y_train = next(iter(train_loader))
    X_test, y_test = next(iter(test_loader))
    X_train, y_train = X_train.cpu().numpy(), y_train.cpu().numpy() # tensor 转为 array 形式)
    X_test, y_test = X_test.cpu().numpy(), y_test.cpu().numpy() # tensor 转为 array 形式)
    X_train = X_train.reshape(X_train.shape[0],784)
    X_test = X_test.reshape(X_test.shape[0],784)
    return X_train, X_test, y_train, y_test

class MLP():
    '''
    使用包含一个输入层，一个隐藏层，一个输出层的 MLP 完成手写数字识别任务
    '''
    def __init__(self, X_train, y_train, lmb=1.0, input_size=784, hidden_size=64, output_size=10):
        '''
        MLP 相关参数初始化
        Args:
            X_train: 训练样本取值
            y_train: 训练样本标签
            lmb: 神经网络正则化参数
            input_size: 输入层神经元个数
            hidden_size: 隐藏层神经元个数
            output_size: 输出层神经元个数 (== num_labels)
        '''
        self.X_train = X_train
        self.y_train = y_train

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lmb = lmb

        # 神经网络参数列表准备，后续进行随机初始化 （关于为什么 + 1，是因为我们在 MLP 中将偏置 b 视为哑神经元）
        self.nn_params = np.array([0.0] * (hidden_size * (input_size + 1) + output_size * (hidden_size + 1)))

    def random_initialize_weights(self, L_in, L_out):
        '''神经网络参数的随机初始化（逐层）
        '''
        eps = np.sqrt(6) / np.sqrt(L_in + L_out)
        max_eps, min_eps = eps, -eps
        W = np.random.rand(L_out, 1 + L_in) * (max_eps - min_eps) + min_eps
        return W

    def param_initialization(self):
        '''神经网络全部权重的随机初始化
        '''
        print('Initializing Neural Network Parameters ...')
        initial_Theta1 = self.random_initialize_weights(self.input_size, self.hidden_size)
        initial_Theta2 = self.random_initialize_weights(self.hidden_size, self.output_size)

        self.nn_params = np.hstack((initial_Theta1.flatten(), initial_Theta2.flatten()))
        pass

    def sigmoid(self, z):
        '''使用 Sigmoid 函数作为 MLP 激活函数
        '''
        return 1.0 / (1.0 + np.exp(-np.asarray(z)))

    def sigmoid_gradient(self, z):
        '''Sigmoid 函数梯度计算
        '''
        g = self.sigmoid(z) * (1 - self.sigmoid(z))
        return g

    def nn_cost_function(self):
        '''神经网络损失计算（包含正则项）
        '''
        Theta1 = self.nn_params[:self.hidden_size * (self.input_size + 1)]
        Theta1 = Theta1.reshape((self.hidden_size, self.input_size + 1))
        Theta2 = self.nn_params[self.hidden_size * (self.input_size + 1):]
        Theta2 = Theta2.reshape((self.output_size, self.hidden_size + 1))

        m = self.X_train.shape[0]

        # 对标签作 one-hot 编码
        one_hot_y = np.zeros((m, self.output_size))
        for idx, label in enumerate(self.y_train):
            one_hot_y[idx, int(label)] = 1

        X = np.hstack([np.ones((m, 1)), self.X_train]) # [batch_size, input_layer_size + 1]
        X = self.sigmoid(np.matmul(X, Theta1.T)).reshape(-1, self.hidden_size) # [batch_size, hidden_layer_size]
        X = np.hstack([np.ones((m, 1)), X]) # [batch_size, hidden_layer_size + 1]
        h = self.sigmoid(np.matmul(X, Theta2.T)).reshape(-1, self.output_size) # [batch_size, num_labels]

        # 计算交叉熵损失项与正则化项
        ce = -one_hot_y * np.log(h) - (1 - one_hot_y) * np.log(1 - h)
        regular = np.sum(np.square(Theta1[:, 1:])) + np.sum(np.square(Theta2[:, 1:]))

        # finally, 含正则化项的损失表达
        J = np.sum(ce) / m + self.lmb * regular / (2 * m)
        return J

    def nn_grad_function(self):
        '''神经网络损失梯度计算
        '''
        Theta1 = self.nn_params[:self.hidden_size * (self.input_size + 1)]
        Theta1 = Theta1.reshape((self.hidden_size, self.input_size + 1))
        Theta2 = self.nn_params[self.hidden_size * (self.input_size + 1):]
        Theta2 = Theta2.reshape((self.output_size, self.hidden_size + 1))

        m = self.X_train.shape[0]

        # 对标签作 one-hot 编码
        one_hot_y = np.zeros((m, self.output_size))
        for idx, label in enumerate(self.y_train):
            one_hot_y[idx, int(label)] = 1

        a1 = np.hstack([np.ones((m, 1)), self.X_train])  # [batch_size, input_layer_size + 1]
        z2 = np.matmul(a1, Theta1.T)
        a2 = self.sigmoid(z2).reshape(-1, self.hidden_size)  # [batch_size, hidden_layer_size]
        a2 = np.hstack([np.ones((m, 1)), a2])  # [batch_size, hidden_layer_size + 1]
        z3 = np.matmul(a2, Theta2.T)
        a3 = self.sigmoid(z3).reshape(-1, self.output_size)  # [batch_size, num_labels]

        Theta1_grad = np.zeros_like(Theta1)
        Theta2_grad = np.zeros_like(Theta2)

        delta_output = a3 - one_hot_y # [batch_size, num_labels]
        delta_hidden = np.matmul(delta_output, Theta2[:, 1:]) * self.sigmoid_gradient(z2)

        Theta1_grad += np.matmul(delta_hidden.T, a1) / m
        Theta2_grad += np.matmul(delta_output.T, a2) / m

        # 包含正则化项 (注意：此时不应考虑偏置)
        Theta1_grad[:, 1:] += self.lmb * Theta1[:, 1:] / m
        Theta2_grad[:, 1:] ++ self.lmb * Theta2[:, 1:] / m

        grad = np.hstack((Theta1_grad.flatten(), Theta2_grad.flatten()))
        return grad

    def train(self, max_iter, learning_rate):
        '''使用梯度下降法训练 MLP
        '''
        print("=" * 60)
        print("Start Training...")
        for i in range(max_iter):
            grad = self.nn_grad_function()
            self.nn_params -= learning_rate * grad
            if i % 10 == 0:
                loss = self.nn_cost_function()
                print('-' * 50)
                print(f"iteration {i}, loss: {loss}")
        print("=" * 60)
        pass

    def forward(self, X):
        '''前向传播
        '''
        Theta1 = self.nn_params[:self.hidden_size * (self.input_size + 1)]
        Theta1 = Theta1.reshape((self.hidden_size, self.input_size + 1))
        Theta2 = self.nn_params[self.hidden_size * (self.input_size + 1):]
        Theta2 = Theta2.reshape((self.output_size, self.hidden_size + 1))

        m = X.shape[0]

        X = np.hstack([np.ones((m, 1)), X])  # [batch_size, input_layer_size + 1]
        X = self.sigmoid(np.matmul(X, Theta1.T)).reshape(-1, self.hidden_size)  # [batch_size, hidden_layer_size]
        X = np.hstack([np.ones((m, 1)), X])  # [batch_size, hidden_layer_size + 1]
        h = self.sigmoid(np.matmul(X, Theta2.T)).reshape(-1, self.output_size)  # [batch_size, num_labels]

        p = np.argmax(h, axis=1)
        return p

if __name__ == '__main__':
    print('Loading data...')
    X_train, X_test, y_train, y_test = load_local_mnist()
    mlp = MLP(X_train=X_train[:2000], y_train=y_train[:2000])
    # 初始化 MLP 权重
    mlp.param_initialization()
    # 模型训练
    mlp.train(max_iter=50, learning_rate=1.0)
    # 模型预测
    pred = mlp.forward(X=X_test[:200])
    print('Test Set Accuracy:', np.mean(pred == y_test[:200]) * 100.0)