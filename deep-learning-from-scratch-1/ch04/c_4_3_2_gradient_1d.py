import numpy as np
import matplotlib.pylab as plt

# 数值微分 numerical differentiation
def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)

# 原始函数: 0.01x^2 + 0.1x
def function_1(x):
    return 0.01 * x**2 + 0.1 * x

# 切线: 0.02x + 0.1
def tangent_line(f, x):
    d = numerical_diff(f, x)
    print(d) # 0.1999999999990898，非常接近 2 了
    y = f(x) - d*x
    return lambda t: d*t + y

plt.xlabel("x")
plt.ylabel("f(x)")

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)

# 在 x = 5 处的切线
tf = tangent_line(function_1, 5)
y2 = tf(x)

plt.plot(x, y)
plt.plot(x, y2)
plt.show()