import matplotlib.pyplot as plt
import os

current_path = os.path.dirname(__file__)
img_path = os.path.join(current_path, '../dataset/lena.png')

img = plt.imread(img_path) # 读入图像

plt.imshow(img)
plt.show()