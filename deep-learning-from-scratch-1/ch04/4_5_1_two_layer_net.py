# coding: utf-8
import sys, os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir) # 临时把 parent_dir 放在 sys.path 里

from common.functions import *
from common.gradient import numerical_gradient
import numpy as np


class TwoLayerNet:

    # input_size: 输入层神经元数，这里为 784 (28 * 28)
    # hidden_size: 隐藏层神经元数
    # output_size: 输出层神经元数，这里为 10 (0-9)
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 初始化权重
        self.params = {}
        # 权重使用符合「高斯分布」的「随机数」进行初始化
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        # 偏置使用 0 进行初始化
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    # 推理用的神经网络
    # x: 输入数据 (图像数据)
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
    
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        return y # 神经网络
        
    # 计算损失函数
    # x: 输入数据 (图像数据)
    # t: 监督数据
    def loss(self, x, t):
        y = self.predict(x)
        
        return cross_entropy_error(y, t) # 输出误差
    
    # 计算识别精度
    # 3.6.3
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1) # 获取概率最高的元素的索引，这里的索引正好和 0-9 数字一一对应
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    # 计算权重参数的梯度
    # x: 输入数据
    # t: 监督数据
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
        
    # 使用「误差反向传播法」高效地计算梯度
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        
        batch_num = x.shape[0]
        
        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        
        dz1 = np.dot(dy, W2.T)
        da1 = sigmoid_grad(a1) * dz1
        grads['W1'] = np.dot(x.T, da1)
        grads['b1'] = np.sum(da1, axis=0)

        return grads