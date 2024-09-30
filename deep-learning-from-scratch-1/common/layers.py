import numpy as np

from common.functions import sigmoid, softmax, cross_entropy_error

class Relu:
    """
    5.5.1 ReLU(Rectified Linear Unit) 层
    """
    def __init__(self):
        self.mask = None

    def forward(self, x):
        """
        x > 0, y = x
        x <= 0, y = 0
        """
        self.mask = (x <= 0) # 把 x 中 <= 0 的地方保存为 True
        out = x.copy()
        out[self.mask] = 0 # True 的下标设置为 0

        return out

    def backward(self, dout):
        """
        x > 0, dy/dx = 1
        x <= 0, dy/dx = 0
        """
        dout[self.mask] = 0 # True 的下标设置为 0
        dx = dout

        return dx


class Sigmoid:
    """
    5.5.2 Sigmoid 层
    """
    def __init__(self):
        self.out = None

    def forward(self, x):
        """
        y = 1 / (1 + e^-x)
        """
        out = sigmoid(x)
        self.out = out # 保存下来，backward 要用到
        return out

    def backward(self, dout):
        """
        dy/dx = y^2 * e^-x
        """
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None
        self.original_x_shape = None # 考虑张量

        self.dW = None
        self.db = None

    def forward(self, x):
        self.original_x_shape = x.shape # 考虑张量
        x = x.reshape(x.shape[0], -1) # 考虑张量
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)

        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape) # 考虑张量

        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # 损失
        self.y = None    # softmax 的输出
        self.t = None    # 监督数据 (one-hot vector)

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 如果 监督数据 是 one-hot 向量
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx