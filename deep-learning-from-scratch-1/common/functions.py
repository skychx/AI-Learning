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

    # 为防止溢出，各输入信号 统一减去 输入信号中的最大值
    x = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

# 交叉熵误差
def cross_entropy_error(y, t):
    """
    交叉熵误差，损失函数的一种 (4.2.4)

    param y: 神经网络的输出
    param t: 监督数据，注意这里 t 是非 one-shot 场景（标签场景）
    returns: 返回一个数字，表示误差
    """
    delta = 1e-7 # 防止 np.log(0) 算出负无穷

    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)
    
    batch_size = y.shape[0] # y 个数

    # 这里的逻辑看起来是针对 mnist 定制的
    # 如果 t 是 one-shot，那么就可以用 -np.sum(t * np.log(y + 1e-7))
    # 计算过程如下：
    # t = [  0,    0,   1,   0,    0,   0,   0,   0,   0,   0]
    # y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
    # 最后其实计算的就是 -ln(0.6)
    #
    # 如果是标签模式，上面的 t 输出的其实是 2
    # **这个 2 又可以表示数字 2，还可以表示为 one-shot 下的下标 2**
    # 从这里就可以看出，如果 y 给的是一个数组，例如 y.shape = (5, 10)
    # 这时：
    # batch_size = 5;
    # np.arange(batch_size) = [0, 1, 2, 3, 4]
    # t = [2, 7, 0, 9, 4], 表示对应的预测为这几个数字，也可以表示 y[n] 中正确的预测概率下标
    # y[np.arange(batch_size), t] 就表示 [y[0,2], y[1,7], y[2,0], y[3,9], y[4,4]]
    # 然后算这些单个数字的 交叉熵误差，然后求和，求平均数
    return -np.sum(
        np.log(
            y[np.arange(batch_size), t] + delta
        )
    ) / batch_size