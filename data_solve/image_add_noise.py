import os
from PIL import Image
import numpy as np


def add_gaussian_noise(image_array, mean=0, std=1):
    """
    为图像添加高斯噪声
    :param image_array: numpy 数组表示的图像
    :param mean: 噪声均值
    :param std: 噪声标准差
    :return: 加噪声后的图像
    """
    noise = np.random.normal(mean, std, image_array.shape)
    noisy_image = image_array + noise
    return noisy_image


def process_image(image_path, output_path):
    """
    处理单个图像
    :param image_path: 输入图像路径
    :param output_path: 输出图像路径
    """
    # 读取图像
    image = Image.open(image_path).convert('L')
    image_array = np.array(image).astype(np.float32)

    # 添加高斯噪声
    noisy_image = add_gaussian_noise(image_array, mean=0, std=1)

    # 获取最大值并进行归一化处理
    max_value = np.max(noisy_image)
    if max_value == 0:
        max_value = 1
    normalized_array = (noisy_image / max_value) * 255

    # 转换为uint8类型
    output_array = normalized_array.astype(np.uint8)

    # 保存处理后的图像
    output_image = Image.fromarray(output_array)
    output_image.save(output_path)


def process_directory(input_dir, output_dir):
    """
    处理目录下的所有图像
    :param input_dir: 输入目录
    :param output_dir: 输出目录
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            process_image(input_path, output_path)


# 设置输入和输出目录路径
input_directory = '/home/jnu/下载/prvamp_jujiao/XL_big_eng/LQ/'
output_directory = '/home/jnu/下载/prvamp_jujiao/XL_big_eng_noise/LQ/'

# 处理目录下的所有图像
process_directory(input_directory, output_directory)