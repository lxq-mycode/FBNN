import os
from PIL import Image

# 定义原始文件夹路径和目标文件夹路径
original_folder_path = '/media/jnu/1eb5aebb-d8ad-4c4a-a83d-4a86dd7d859a/1000张/two_maoboli/sp'
target_folder_path = '/media/jnu/1eb5aebb-d8ad-4c4a-a83d-4a86dd7d859a/1000张/two_maoboli/sp_512'

# 创建目标文件夹
os.makedirs(target_folder_path, exist_ok=True)

# 遍历原始文件夹中的所有图片文件
for file in os.listdir(original_folder_path):
    file_path = os.path.join(original_folder_path, file)

    # 确保当前路径是一个文件
    if os.path.isfile(file_path):
        # 打开图片文件
        img = Image.open(file_path)

        # 获取原图的宽高
        width, height = img.size

        # 计算裁剪区域的左上角和右下角坐标
        left = (width - 512) // 2
        upper = (height - 512) // 2
        right = left + 512
        lower = upper + 512

        # 裁剪图片为512x512大小
        img_cropped = img.crop((left, upper, right, lower))

        # 保存裁剪后的图片到目标文件夹
        target_file_path = os.path.join(target_folder_path, file)
        img_cropped.save(target_file_path)

print("图片裁剪完成！")
