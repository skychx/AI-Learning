import sys, os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir) # 临时把 parent_dir 放在 sys.path 里

import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

print('x_train.shape:', x_train.shape) # (60000, 784)
print('t_train.shape:', t_train.shape) # (60000, 10)


train_size = x_train.shape[0]
batch_size = 10

# 从 60000 个里随机挑选 10 个，这里的 batch_mask 为数组下标
batch_mask = np.random.choice(train_size, batch_size)

x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

print('batch_mask:', batch_mask)
print('x_batch:\n', x_batch)
print('t_batch:\n', t_batch)
