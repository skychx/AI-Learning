# coding: utf-8
import sys, os

current_path = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_path, '..'))
sys.path.append(parent_dir) # 临时把 parent_dir 放在 sys.path 里

import numpy as np
import matplotlib.pylab as plt

from common.functions import sigmoid, softmax

# 恒等函数
def identity_function(x):
    return x

def init_network():
    _network = {}

    # 第 1 层 (隐藏层)，3 个神经元
    _network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    _network['b1'] = np.array([0.1, 0.2, 0.3])
    # 第 2 层 (隐藏层)，3 个神经元
    _network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    _network['b2'] = np.array([0.1, 0.2])
    # 第 3 层 (输出层)，2 个神经元
    _network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    _network['b3'] = np.array([0.1, 0.2])

    return _network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    # A: 输出信号
    # X: 输入信号，W: 权重，B: 偏置
    # A = X * W + B
    # Z = sigmoid(X * W + B)

    # 隐藏层 1
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)

    # 隐藏层 2
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)

    # 输出层
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3) # 此章节还未学习 softmax 函数，所以输出和不为 1

    return y

network = init_network()

# 输入层，2 个神经元
x = np.array([1.0, 0.5])
y = forward(network, x)

print(y) # [0.31682708 0.69627909]