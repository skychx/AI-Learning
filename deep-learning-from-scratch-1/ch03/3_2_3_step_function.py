import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    """
    阶跃函数, x 大于 0, 返回 1; 否则返回 0
    """
    return np.array(x > 0, dtype=int)

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)

plt.plot(x, y)
plt.ylim(-0.1, 1.1) # 指定y轴的范围

plt.show()