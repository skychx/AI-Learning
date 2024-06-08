import numpy as np

def sigmoid(x):
    """
    sigmoid 激活函数 (3.2.1)，是一条平滑的曲线

    param x: 神经网络某一层的输入
    returns: 返回一个数组
    """

    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


def softmax(x):
    """
    分类函数 (3.5.1)，典型特征是输出值总和为 1

    param x: 神经网络的输出
    returns: 返回一个数组
    """

    x = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

# 交叉熵误差
def cross_entropy_error(y, t):
    """
    交叉熵误差，损失函数的一种 (4.2.4)

    param y: 神经网络的输出
    param t: 监督数据，注意这里 t 是非 one-shot 场景
    returns: 返回一个数字，表示误差
    """
    delta = 1e-7 # 防止 np.log(0) 算出负无穷

    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)
    
    batch_size = y.shape[0]

    return -np.sum(
        np.log(
            y[np.arange(batch_size), t] + delta
        )
    ) / batch_size