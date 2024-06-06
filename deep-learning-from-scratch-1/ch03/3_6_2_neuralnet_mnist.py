# coding: utf-8
import sys, os

current_path = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_path, '..'))
sys.path.append(parent_dir) # 临时把 parent_dir 放在 sys.path 里

import time
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax

start_time = time.time()

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

# 读取「预先训练」到的权重和偏置参数
def init_network():
    pkl_path = os.path.join(current_path, 'sample_weight.pkl')

    with open(pkl_path, 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

x, t = get_data()
network = init_network()
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y) # 获取概率最高的元素的索引，这里的索引正好和 0-9 数字一一对应
    if p == t[i]:
        accuracy_cnt += 1

end_time = time.time()

print("Accuracy: " + str(float(accuracy_cnt) / len(x))) # 0.9352，准确率 93.52%
print("time cost: ", round((end_time - start_time) * 1000, 2), "ms") # 241ms
