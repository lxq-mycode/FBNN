from PIL import Image
import numpy as np


def count_pixels_above_threshold(image_path, threshold=50):
    # 打开图像并转换为灰度图像
    image = Image.open(image_path).convert('L')

    # 将图像转换为NumPy数组
    image_array = np.array(image)

    # 计算像素值大于阈值的像素点数量
    count = np.sum(image_array > threshold)

    return count


# 示例：计算 'example.png' 中像素值大于50的像素点数量
image_path = '/media/jnu/SUCCESS/data_7_5/hanzi/net/LQ/5.png'
threshold = 2
count = count_pixels_above_threshold(image_path, threshold)

print(f"像素值大于 {threshold} 的像素点数量: {count}")