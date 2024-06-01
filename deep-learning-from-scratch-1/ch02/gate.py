import numpy as np

def AND(x1, x2):
  x = np.array([x1, x2])
  w = np.array([0.5, 0.5]) # 权重
  b = -0.7                 # 偏置
  tmp = np.sum(x * w) + b

  if tmp <= 0:
    return 0
  else:
    return 1


def NAND(x1, x2):
  x = np.array([x1, x2])
  # 和 AND 相比，权重 和 偏置都取反
  w = np.array([-0.5, -0.5]) # 权重
  b = 0.7                    # 偏置
  tmp = np.sum(x * w) + b

  if tmp <= 0:
    return 0
  else:
    return 1

def OR(x1, x2):
  x = np.array([x1, x2])
  w = np.array([0.5, 0.5]) # 权重
  b = -0.2                 # 偏置
  tmp = np.sum(x * w) + b

  if tmp <= 0:
    return 0
  else:
    return 1


def XOR(x1, x2):
  s1 = NAND(x1, x2)
  s2 = OR(x1, x2)
  y = AND(s1, s2)
  return y