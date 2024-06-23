import numpy as np

# 交叉熵误差
def cross_entropy_error(y, t):
    delta = 1e-7 # 防止 np.log(0) 出现无限小的场景
    return -np.sum(t * np.log(y + delta))

# 这里的 t 为 one-shot
t =  [  0,    0,   1,   0,    0,   0,   0,   0,   0,   0]
y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]

# 数学上实际为 -ln(0.6) 和 -ln(0.1)
print('y1 :', cross_entropy_error(np.array(y1), np.array(t))) # 0.510825457099338
print('y2 :', cross_entropy_error(np.array(y2), np.array(t))) # 2.302584092994546