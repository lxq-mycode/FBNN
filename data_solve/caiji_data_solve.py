import tifffile as tiff
from PIL import Image
import os

# 设置输入和输出路径
input_tif_path = '/media/jnu/1eb5aebb-d8ad-4c4a-a83d-4a86dd7d859a/caiji_dataset/9_25/Video10.tif'  # 输入的tif文件路径
output_folder = '/media/jnu/1eb5aebb-d8ad-4c4a-a83d-4a86dd7d859a/caiji_dataset/slove_data/LQ'  # 保存输出png的文件夹路径


# 确保输出文件夹存在，不存在则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 读取tif文件
tif_images = tiff.imread(input_tif_path)

# 检查图片数量
num_images = tif_images.shape[0] if len(tif_images.shape) > 2 else 1  # 单图像时 shape 可能不一样

# 循环保存每一张图片
for i in range(1000):
    img = tif_images[i] if num_images > 1 else tif_images  # 多张或单张图片处理
    img = Image.fromarray(img)  # 将NumPy数组转为PIL图像
    img = img.convert('L')  # 确保图像为灰度模式
    img.save(os.path.join(output_folder, f'{i+9001}.png'))  # 保存为png文件
    print(f'Saved image {i+9001}.png')


print('All images extracted and saved successfully.')
