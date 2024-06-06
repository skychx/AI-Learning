# coding: utf-8
import sys, os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir) # 临时把 parent_dir 放在 sys.path 里

import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img)) # 把保存为 NumPy 数组的图像数据转换为 PIL 用 的数据对象
    pil_img.show()

# (训练图像 , 训练标签)，(测试图像，测试标签)
# normalize: 是否将输入图像「正则化 (normalization)」为 0.0~1.0 的值
# flatten: 是否展开输入图像 (变成一维数组)
# one_hot_label: 是否将标签保存为 one-hot 表示
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=False, flatten=True)

img = x_train[0]
label = t_train[0]
print('label:', label) # 5

print('img shape:', img.shape) # (784,)

img = img.reshape(28, 28)      # 第一张训练图片，reshape() 方法的参数指定原来的大小
print('img shape:', img.shape) # (28, 28)

img_show(img)