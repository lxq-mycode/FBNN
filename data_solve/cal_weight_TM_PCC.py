import h5py
import numpy as np
import torch
from scipy.stats import pearsonr

# 步骤1: 加载.mat文件中的权值矩阵
with h5py.File('/media/jnu/data2/data_mat/AmpSLM_64x64/A_prVAMP.mat', 'r') as file1:
    data1 = file1['A'][:]  # 假设数据存储在名为'A'的变量中

# 检查data1的数据类型
print("Data type of data1:", data1.dtype)

# 将结构化数组转换为复数数组
data1_complex = data1['real'] + 1j * data1['imag']
# data1_complex = data1['real']
# 计算复数矩阵的强度值
data1_magnitude = np.abs(data1_complex)

# 翻转矩阵的形状
# data1_magnitude = data1_magnitude.transpose()  # 使用 transpose 方法
prvamp_mean=np.mean(data1_magnitude)
# 步骤2: 从PyTorch模型中提取权值矩阵
model_path =  '/media/jnu/data2/model_Linear/Linear_net_6464_one_linear/model_41.pth'
model = torch.load(model_path)
state_dict = model.state_dict()
weights_model = None

for layer_name, layer_params in state_dict.items():
    if 'weight' in layer_name:
        weights_model = layer_params.cpu().numpy()
        break  # 假设我们只需要第一个权重矩阵
weights_model=np.abs(weights_model)
weights_model_mean=np.mean(weights_model)
# 步骤3: 计算两个权值矩阵之间的PCC
weights_model=prvamp_mean/weights_model_mean*weights_model
# 确保两个矩阵的形状相同，否则无法计算相关系数
if weights_model.shape == data1_magnitude.shape:
    corr_coef = np.corrcoef(data1_magnitude.flatten(),weights_model.flatten())[0,1]
    print(f"Pearson Correlation Coefficient: {corr_coef}")
else:
    print("The shapes of the two matrices do not match, cannot calculate PCC.")


