# import cv2
# import numpy as np
# from PIL import Image
# # test_matfocus_noise_6464_path_plus = '/media/jnu/data1/focus_test/net_2/GT'
# test_matfocus_noise_6464_path_plus = '/media/jnu/SUCCESS/data_5_29/jujiao/netmathjujiao/jujiaoyuce/testmodel_clear1'
#
#
#
#
# test_noise = []
# test_clear = []
#
# for idx in range(1, 6):
#     # get the match imgs of clears and noises
#     n_path = ('%s/%d.png' % (test_matfocus_noise_6464_path_plus, idx))
#     img = cv2.imread(n_path, cv2.IMREAD_GRAYSCALE)
#     # append clear imgs and noise imgs
#     # 归一化
#     # img = img / np.max(img)
#     img = (img -np.min(img))/ (np.max(img)-np.min(img))
#     # 将图像转换为8位整型
#     img = np.uint8(img * 255)
#     img = Image.fromarray(img)
#     img.save(n_path)

import cv2
import numpy as np
from PIL import Image

test_matfocus_noise_6464_path_plus = '/media/jnu/1eb5aebb-d8ad-4c4a-a83d-4a86dd7d859a/one_paper_data/paper_data/TM/test/JT_net/LQ/testmodel_clear1'
# test_matfocus_noise_6464_path_plus = '/media/jnu/SUCCESS/data_7_5/hanzi_1/net/yuce/testmodel_clear1'
for idx in range(1, 6):
    # 获取图像路径
    n_path = ('%s/%d.png' % (test_matfocus_noise_6464_path_plus, idx))
    img = cv2.imread(n_path, cv2.IMREAD_GRAYSCALE)

    # 归一化到0到1之间
    # img = (img - np.min(img)) / (np.max(img) - np.min(img))

    # 取平方根
    # img = np.square(img)

    # 重新归一化到0到255之间
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255

    # 将图像转换为8位整型
    img = np.uint8(img)

    # 保存处理后的图像
    img = Image.fromarray(img)
    img.save(n_path)