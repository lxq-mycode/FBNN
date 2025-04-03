import cv2
import numpy as np
from PIL import Image


test_clear_path = '/home/jnu/文档/L1_sp/true'
test_noise = '/home/jnu/文档/L1_sp/yuce'



pccs = []  # 初始化一个空列表来存储相关系数

for idx in range(1, 7):
    c_path = ('%s/%d.png' % (test_clear_path, idx))
    n_path = ('%s/%d.png' % (test_noise, idx))
    image1 = Image.open(c_path)
    image2 = Image.open(n_path)
    img1 = np.array(image1).flatten()
    img2 = np.array(image2).flatten()

    # 计算相关系数
    corr_coef = np.corrcoef(img1, img2)[0, 1]
    pccs.append(corr_coef)  # 将相关系数添加到列表中
    print("第", idx, "张图片的相关系数是", corr_coef)

# 计算所有相关系数的平均值
average_pcc = np.mean(pccs)
print("所有图片对之间的平均相关系数是", average_pcc)



