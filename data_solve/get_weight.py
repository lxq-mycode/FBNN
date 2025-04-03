# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# # path = '/media/jnu/data2/model6464to256256/final_test_minist1/model_21.pth'
# path = '/media/jnu/data2/model_9_27/Linear_3L_true/model_31.pth'
# # '/media/jnu/data1/model//'
# # path = '/media/jnu/data1/model/for_model10000/model_41.pth'
#
# # path = '/media/jnu/data2/model_new/co/model_41.pthnv_Linear_net36464/model_41.pth'
# model =  torch.load(path)
# state_dict =model.state_dict()
# weights = None
#
#
#
# # weights =weights.to('cuda:1')
# for layname,laypaeams in state_dict.items():
#     if 'weight' in layname:
#         weights =laypaeams.cpu().numpy()
#         # 计算权值矩阵的均值和标准差
#         mean = np.mean(weights)
#         std = np.std(weights)
#         np.save('/media/jnu/data2/10_2_get_weight_TM/weights.npy',weights)
#         # 绘制高斯分布曲线
#         x = np.linspace(mean - 3 * std, mean + 3 * std, 100)
#         y = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(- (x - mean) ** 2 / (2 * std ** 2))
#         plt.plot(x, y, color='r')
#
#         # 绘制直方图
#         plt.hist(weights.flatten(), bins=50, density=True)
#         plt.savefig('/home/jnu/下载/TM_45000.jpg')
#         # 显示图形
#         plt.show()
#
#
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

# 加载模型
path = '/media/jnu/data2/model_9_27/Linear_3L_true/model_31.pth'
model = torch.load(path)
state_dict = model.state_dict()
weights = None

# 遍历模型中的权重
for layname, layparams in state_dict.items():
    if 'weight' in layname:
        weights = layparams.cpu().numpy()
        # 计算权值矩阵的均值和标准差
        mean = np.mean(weights)
        std = np.std(weights)

        # # 保存为.npy文件
        # np.save('/media/jnu/data2/10_2_get_weight_TM/weights.npy', weights)

        # 保存为.mat文件
        # savemat('/media/jnu/data2/10_2_get_weight_TM/weights.mat', {'weights': weights})

        # 绘制高斯分布曲线
        x = np.linspace(mean - 3 * std, mean + 3 * std, 100)
        y = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(- (x - mean) ** 2 / (2 * std ** 2))
        plt.plot(x, y, color='r')

        # 绘制直方图
        plt.hist(weights.flatten(), bins=50, density=True)
        plt.savefig('/home/jnu/下载/TM_45000.jpg')
        # 显示图形
        plt.show()