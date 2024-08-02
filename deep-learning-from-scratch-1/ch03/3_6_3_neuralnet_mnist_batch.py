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


def init_network():
    pkl_path = os.path.join(current_path, 'sample_weight.pkl')

    with open(pkl_path, 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1) + b1 # 批处理不影响最后的结果输出，原因可见 图3-27 的线性代数推导
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()

batch_size = 100 # 批处理，一次性处理 100 个图像
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size] # 取 [i, i+batch_size) 的数据
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

end_time = time.time()

print("Accuracy: " + str(float(accuracy_cnt) / len(x))) # 0.9352
print("time cost: ", round((end_time - start_time) * 1000, 2), "ms") # 85ms
