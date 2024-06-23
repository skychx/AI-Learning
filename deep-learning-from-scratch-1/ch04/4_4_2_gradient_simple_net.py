import sys, os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir) # 临时把 parent_dir 放在 sys.path 里

import numpy as np

from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        # 2x3 的权重参数 W
        self.W = np.random.randn(2,3) # 用高斯分布进行初始化

    # 只做了一层网络，复习可参考 3_4_3_forward_network.py
    # 注意这里为了简化，忽略了偏置 b
    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)

        # 计算损失函数误差
        loss = cross_entropy_error(y, t)

        return loss

x = np.array([0.6, 0.9]) # 输入数据
t = np.array([0, 0, 1]) # 正确解标签

net = simpleNet()

f = lambda w: net.loss(x, t)

# dW 就是神经网络的梯度
# 这里所说的梯度是指「损失函数」关于「权重参数」的梯度
dW = numerical_gradient(f, net.W)

print(dW)