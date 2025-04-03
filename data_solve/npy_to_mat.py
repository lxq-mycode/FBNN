import numpy as np
from scipy.io import savemat


def npy_to_mat(npy_filename, mat_filename, variable_name='data'):
    """
    将.npy文件转换为.mat文件

    参数:
    npy_filename (str): 输入的.npy文件名。
    mat_filename (str): 输出的.mat文件名。
    variable_name (str): 在.mat文件中保存的变量名，默认为'data'。
    """
    # 加载.npy文件
    data = np.load(npy_filename)

    # 将数据保存为.mat文件
    savemat(mat_filename, {variable_name: data})


# 使用示例
npy_filename = "/media/jnu/data2/xwx/GS/64/64_64.npy"   # 替换为你的.npy文件名
mat_filename = '/media/jnu/data2/xwx/GS/64/64TM.mat'  # 你想要创建的.mat文件名
variable_name = 'TM'  # 你希望在.mat文件中使用的变量名

npy_to_mat(npy_filename, mat_filename, variable_name)

