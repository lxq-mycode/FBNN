import os
import numpy as np
import cv2
import torch

def l1_loss(predictions, targets):
    # 读取图像并转换为灰度图
    clear_data = cv2.imread(predictions, cv2.IMREAD_GRAYSCALE)
    noise_data = cv2.imread(targets, cv2.IMREAD_GRAYSCALE)

    # 检查图像是否正确读取
    if clear_data is None or noise_data is None:
        print(f"Error: 图像文件未找到或路径错误。文件：{predictions} 或 {targets}")
        return None

    # 将图像数据转换为 numpy 数组
    clear_data = np.array(clear_data, dtype=np.float32)
    noise_data = np.array(noise_data, dtype=np.float32)

    # 将 numpy 数组转换为 PyTorch 张量
    clear_tensor = torch.from_numpy(clear_data).unsqueeze(0)  # 添加一个批次维度
    noise_tensor = torch.from_numpy(noise_data).unsqueeze(0)

    # 指定设备，如果不支持 CUDA，则使用 'cpu'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clear_tensor = clear_tensor.to(device)
    noise_tensor = noise_tensor.to(device)

    # 创建 L1 损失函数
    loss_function = torch.nn.L1Loss()

    # 计算损失
    loss = loss_function(clear_tensor, noise_tensor)

    return loss

def calculate_losses(clear_path, noise_path):
    clear_files = set(os.listdir(clear_path))
    noise_files = set(os.listdir(noise_path))

    # 取交集，找到两个目录下都有的文件
    common_files = clear_files & noise_files

    if not common_files:
        print("Error: 没有找到相同名字的图片文件。")
        return

    for file_name in common_files:
        clear_file = os.path.join(clear_path, file_name)
        noise_file = os.path.join(noise_path, file_name)
        loss_value = l1_loss(clear_file, noise_file)
        if loss_value is not None:
            print(f"{file_name}: {loss_value.item() / 255}")

# 假设 test_clear_path 和 test_noise_2000_path 是图像文件的路径
test_clear_path = '/media/jnu/SUCCESS/paper_data/TM/test/JT_net/HQ'
test_noise_2000_path = '/media/jnu/SUCCESS/paper_data/TM/TM_cross/prvamp_net/yuce/testmodel_clear1'

# 调用函数并打印结果
calculate_losses(test_clear_path, test_noise_2000_path)