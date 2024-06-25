from functions import sigmoid

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