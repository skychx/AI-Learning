# coding: utf-8
import numpy as np
import matplotlib.pylab as plt

# 用来求某个函数的偏导
# f: 要求偏导的函数
# x: 函数参数列表
def _numerical_gradient_no_batch(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 还原原始值
        
    return grad # 梯度（gradient）

# 求梯度的函数，支持批量
def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)
        
        for i, x in enumerate(X):
            grad[i] = _numerical_gradient_no_batch(f, x)
        
        return grad

# f = x0^2 + x1^2 + ... + xn^2
# 这里的 x 是一个数组
def function_2(x):
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)


if __name__ == '__main__':
    x0 = np.arange(-2, 2.5, 0.25)
    x1 = np.arange(-2, 2.5, 0.25)
    X0, X1 = np.meshgrid(x0, x1)
    
    X0 = X0.flatten()
    X1 = X1.flatten()

    # 上面的用法构建出 X0 和 X1 组合起的参数列表

    # 求出在各个 (x0, x1) 点上求出的梯度
    grad = numerical_gradient(function_2, np.array([X0, X1]).T).T

    plt.figure()
    # quiver 用来画箭头
    plt.quiver(X0, X1, -grad[0], -grad[1], angles="xy",color="#666666")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.draw()
    plt.show()