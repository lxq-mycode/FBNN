import random
import PIL.Image as Image
import numpy as np
import os
import gc
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.image as mpimg  # mpimg 用于读取图片
import cv2

# 设置均值
def gass():
    # 设置均值
    mean = (0, 0)
    # 精度矩阵，协方差矩阵的倒数
    alpha = [[0.5, 0.],
             [0., 0.5]]
    # 初始矩阵256*256*2
    g = np.random.multivariate_normal(mean, alpha,(128*128, 100*100))
    '''
    从多元正态分布中随机抽取样本的函数:
    mean：mean是多维分布的均值维度为1
    cov：协方差矩阵，注意：协方差矩阵必须是对称的且需为半正定矩阵
    size：指定生成的正态分布矩阵的维度（例：若size=(1, 1, 2)，则输出的矩阵的shape即形状为 1X1X2XN（N为mean的长度））
    check_valid：这个参数用于决定当cov即协方差矩阵不是半正定矩阵时程序的处理方式，它一共有三个值：warn，raise以及ignore。
    当使用warn作为传入的参数时，如果cov不是半正定的程序会输出警告但仍旧会得到结果；当使用raise作为传入的参数时，
    如果cov不是半正定的程序会报错且不会计算出结果；当使用ignore时忽略这个问题即无论cov是否为半正定的都会计算出结果。3种情况的console打印结果如下：
    '''
    g = g.astype(np.float32)
    # 分成两个矩阵，相位矩阵和振幅矩阵
    x = (g[:,:, 0])
    y = (g[:,:, 1])
    z = x + 1j * y

    s_z = np.linalg.svd(z,1,0)
    z1 = z / s_z[0]
    np.save("/media/jnu/data2/xwx/GS/100/128_100.npy", z1)


    # x1, y1即为所需要的TM
    return z1
    # 验证TM
    # plt.show()
    # plt.scatter(x1, y1, s = 1)
    # plt.show()


## 读取npy文件作为tm
def load_tm_from_npy(npy_path):
    return np.load(npy_path)


if __name__ == '__main__':
    # tm = gass()

    npy_path = "/media/jnu/data2/xwx/GS/80/128_80.npy"  # 修改为你的单个npy文件路径
    root_path = os.path.join("/media/jnu/data2/xwx/GS/80/face/our_face_HQ")
    new_folder = "/media/jnu/data2/xwx/GS/80/face/our_face_LQ"

    # 读取npy文件作为tm
    tm = load_tm_from_npy(npy_path)
    # 判断文件夹是否存在，不存在则创建
    # tm = gass()
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
    filename_list = os.listdir(root_path)
    for i in range(0, len(filename_list)):
        im_path = ('%s//%d.png' % (root_path, i+1))
        im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)

        data = im.reshape(1, -1)
        data = data.T  # 按行展开并转置

        result = np.dot(tm, data)  # 矩阵相乘
        result = result.T
        result = result.reshape(128, 128)

        result1 = (np.abs(result)**2)/np.max(np.abs(result)**2)*255

        # 确保结果图像为uint8类型，以便正确保存为灰度图像
        result1 = result1.astype(np.uint8)

        new_im = Image.fromarray(result1, mode='L')  # 明确指定模式为'L'（灰度）
        new_im.save(os.path.join(new_folder, str(i + 1) + '.png'))  # 分别命名图片

#
#
#
#
#
#
#
#



#使用GPU来生成TM





# import torch
# import torch.nn.functional as F
# import numpy as np
# import os
# from PIL import Image
# import cv2
#
# # 设置均值
# def gass():
#     # 设置均值
#     mean = torch.tensor([0, 0], dtype=torch.float32, device='cuda')
#     # 精度矩阵，协方差矩阵的倒数
#     alpha = torch.tensor([[0.8, 0.],
#                           [0., 0.8]], dtype=torch.float32, device='cuda')
#     # 初始矩阵256*256*2
#     g = torch.distributions.multivariate_normal.MultivariateNormal(mean, alpha).sample((128*128, 64*64))
#     g = g.to(torch.float32)
#     # 分成两个矩阵，相位矩阵和振幅矩阵
#     x = g[:, :, 0]
#     y = g[:, :, 1]
#     z = x + 1j * y
#
#     # SVD分解
#     s_z = torch.svd(z)[1]
#     z1 = z / s_z[0]
#     torch.save(z1.cpu(), "/media/jnu/data2/xwx/GS/64/64_64.npy")
#
#     # x1, y1即为所需要的TM
#     return z1
#
# ## 读取npy文件作为tm
# def load_tm_from_npy(npy_path):
#     return torch.load(npy_path).to('cuda')
#
# if __name__ == '__main__':
#     # tm = gass()
#
#     npy_path = "/media/jnu/data2/xwx/GS/32/32_tm_128.npy"  # 修改为你的单个npy文件路径
#     root_path = os.path.join("/media/jnu/data2/xwx/dataset/32_32/new_data/HQ")
#     new_folder = "/media/jnu/data2/xwx/dataset/32_32/new_data/SP"
#     # 读取npy文件作为tm
#     # tm = load_tm_from_npy(npy_path)
#     # 判断文件夹是否存在，不存在则创建
#     tm = gass()
#     if not os.path.exists(new_folder):
#         os.makedirs(new_folder)
#     filename_list = os.listdir(root_path)
#     for i in range(0, len(filename_list)):
#         im_path = ('%s//%d.png' % (root_path, i+1))
#         im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
#         im_tensor = torch.tensor(im, dtype=torch.float32, device='cuda').view(-1, 1)
#
#         # 将实数张量转换为复数张量
#         im_tensor_complex = torch.view_as_complex(torch.stack((im_tensor, torch.zeros_like(im_tensor)), dim=-1))
#
#         result = torch.matmul(tm, im_tensor_complex)
#         result = result.view(128, 128)
#
#         result1 = (torch.abs(result)**2)/torch.max(torch.abs(result)**2)*255
#
#         # 将结果转换回CPU并转换为numpy数组
#         result1 = result1.cpu().numpy().astype(np.uint8)
#
#         new_im = Image.fromarray(result1, mode='L')  # 明确指定模式为'L'（灰度）
#         new_im.save(os.path.join(new_folder, str(i + 1) + '.png'))  # 分别命名图片
#
