import numpy as np

def AND(x1, x2):
  """
  与门 仅在两个输入均为 1 时输出 1, 其他时候则输出 0

  x1 x2 y
  -------
  0  0  0
  1  0  0
  0  1  0
  1  1  1
  """
  x = np.array([x1, x2])
  w = np.array([0.5, 0.5]) # 权重
  b = -0.7                 # 偏置
  tmp = np.sum(x * w) + b

  if tmp <= 0:
    return 0
  else:
    return 1


def NAND(x1, x2):
  """
  与非门 就是颠倒 与门 的输出

  x1 x2 y
  -------
  0  0  1
  1  0  1
  0  1  1
  1  1  0
  """
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
  """
  或门 就是只要有一个输入信号是 1, 输出就为 1

  x1 x2 y
  -------
  0  0  0
  1  0  1
  0  1  1
  1  1  1
  """
  x = np.array([x1, x2])
  w = np.array([0.5, 0.5]) # 权重
  b = -0.2                 # 偏置
  tmp = np.sum(x * w) + b

  if tmp <= 0:
    return 0
  else:
    return 1


def XOR(x1, x2):
  """
  异或门 就是仅当 x1 或 x2 中的一方为 1 时, 才会输出 1 

  x1 x2 y
  -------
  0  0  0
  1  0  1
  0  1  1
  1  1  0
  """
  # 通过叠加多层感知机可以实现 XOR
  # x1 x2 s1 s2 y
  # -------------
  # 0  0  1  0  0
  # 1  0  1  1  1
  # 0  1  1  1  1
  # 1  1  0  1  0
  s1 = NAND(x1, x2)
  s2 = OR(x1, x2)
  y = AND(s1, s2)
  return y