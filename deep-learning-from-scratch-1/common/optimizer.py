# coding: utf-8
import numpy as np

class SGD:
    """
    6.1.2 随机梯度下降法
    """

    # lr: leanring rate（学习率）
    def __init__(self, lr=0.01):
        self.lr = lr

    # params: 权重参数 object
    # grads:  梯度 object
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class Momentum:
    """
    6.1.4 Momentum SGD
    """

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr             # 学习率
        self.momentum = momentum # 相当于摩擦力
        self.v = None            # 速度 map

    def update(self, params, grads):
        # 默认速度为 0
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        # 算法本身
        for key in params.keys():
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key] 
            params[key] += self.v[key]


class AdaGrad:
    """
    6.1.5 AdaGrad

    AdaGrad 会记录过去所有梯度的平方和。因此，学习越深入，更新的幅度就越小。
    实际上，如果无止境地学习，更新量就会变为 0, 完全不再更新
    """

    def __init__(self, lr=0.01):
        self.lr = lr  # 学习率
        self.h = None # 梯度平方和 map
        
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
            
        for key in params.keys():
            self.h[key] += grads[key] * grads[key] # 更新平方和
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class RMSprop:
    """
    6.1.5 RMSprop

    http://zh.gluon.ai/chapter_optimization/rmsprop.html
    """

    def __init__(self, lr=0.01, decay_rate = 0.99):
        self.lr = lr                  # 学习率
        self.decay_rate = decay_rate  # 遗忘率
        self.h = None                 # 梯度平方和 map
        
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
            
        for key in params.keys():
            # 指数移动平均
            self.h[key] *= self.decay_rate
            self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class Adam:

    """
    6.1.6 Adam

    融合了 Momentum 和 AdaGrad 的方法，还能进行超参数的 “偏置校正”
    
    http://arxiv.org/abs/1412.6980v8
    http://zh.gluon.ai/chapter_optimization/adam.html
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr        # 学习率
        self.beta1 = beta1  # momentum 系数 β1
        self.beta2 = beta2  # momentum 系数 β2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        
        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)
        
        for key in params.keys():
            #self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            #self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)