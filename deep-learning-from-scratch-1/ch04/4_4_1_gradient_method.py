# coding: utf-8
import importlib
import numpy as np
import matplotlib.pylab as plt

# 使用 importlib 动态导入数字开头的文件
# 并不是很好的办法，只是用数字表示序号更方便一些
numerical_gradient = importlib.import_module('4_3_3_gradient_2d').numerical_gradient

# 梯度下降法 (gradient descent method)
# 沿着梯度方向步进，识图找到函数的最小值（或相对最小值）
# lr 这个步进值，就叫 学习率 (learning rate)，过大过小都不行
# 这种参数就叫「超参数」，是人工设定的，需要试，也就是所谓的「调参」
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append( x.copy() )

        grad = numerical_gradient(f, x)
        x -= lr * grad # 求完梯度就开始步进

    return x, np.array(x_history)


def function_2(x):
    return x[0]**2 + x[1]**2

init_x = np.array([-3.0, 4.0])

lr = 0.1
step_num = 20

# 从 (-3, 4) 开始梯度下降 20 次
x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)

print('x:\n', x)
print('x_history:\n', x_history)

plt.plot( [-5, 5], [0,0], '--b')
plt.plot( [0,0], [-5, 5], '--b')
plt.plot(x_history[:,0], x_history[:,1], 'o')

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()