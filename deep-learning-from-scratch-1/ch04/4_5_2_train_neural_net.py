# coding: utf-8
import sys, os
import time

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir) # 临时把 parent_dir 放在 sys.path 里

import importlib
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist

TwoLayerNet = importlib.import_module('4_5_1_two_layer_net').TwoLayerNet

start_time = time.time()

# (训练图像 , 训练标签)，(测试图像，测试标签)
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 超参数
iters_num = 10000 # 步进数，也是循环数
learning_rate = 0.1 # 步进值

train_size = x_train.shape[0] # 这里有 60000 个数据
batch_size = 100 # 每次从 60000 个训练数据中随机取出 100 个数据

train_loss_list = []
train_acc_list = [] # 这个是存历史数据画图的
test_acc_list = []  # 这个是存历史数据画图的

# 平均每个 epoch 的重复次数，这里就是 600
# 这里粗略的认为每 600 次循环可以遍历完毕所有的测试数据
iter_per_epoch = max(train_size / batch_size, 1)


for i in range(iters_num):
    # print('i:', i, time.time())
    # 获取 mini-batch
    batch_mask = np.random.choice(train_size, batch_size) # 注意这里是随机选的
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 计算梯度
    # grad = network.numerical_gradient(x_batch, t_batch) # 算 1 次耗时 70 s
    grad = network.gradient(x_batch, t_batch) # 算 10000 次耗时 61.52 s
    
    # 随机梯度下降法(SGD) 更新参数
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    # 记录学习过程
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    # 计算每个 epoch 的识别精度
    # 循环总次数为 10000，iter_per_epoch 为 600
    # 所以每 600 次算一下识别精度
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train) # 训练数据的计算精度
        test_acc = network.accuracy(x_test, t_test) # 测试数据的计算精度
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        # 从 0.11 增长到 0.94，并且，这两个识别精度基本上没有差异
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

end_time = time.time()

print("time cost:", round((end_time - start_time), 2), "s")

# 绘图
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()

