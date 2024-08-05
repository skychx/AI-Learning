import numpy as np

# 均方差
def mean_squared_error(y, t):
    # y 表示神经网络的输出数据，t 表示监督数据，用来计算均方误差
    return 0.5 * np.sum((y-t)**2)

t =  [  0,    0,   1,   0,    0,   0,   0,   0,   0,   0]
y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]

print('y1 :', mean_squared_error(np.array(y1), np.array(t))) # 0.0975
print('y2 :', mean_squared_error(np.array(y2), np.array(t))) # 0.5975