import os
from PIL import Image


def convert_bmp_to_png_and_delete(folder_path):
    # 遍历目录下的所有文件
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.bmp'):
            # 构建完整的文件路径
            file_path = os.path.join(folder_path, filename)

            try:
                # 尝试打开图片
                with Image.open(file_path) as img:
                    # 如果图片不是灰度模式，则转换为灰度模式
                    if img.mode != 'L':
                        img = img.convert('L')

                    # 构建输出文件的路径，直接覆盖原文件
                    output_path = os.path.splitext(file_path)[0] + '.png'

                    # 保存新图片为PNG格式
                    img.save(output_path, 'PNG')
                    print(f'Converted {file_path} to {output_path}')

                    # 删除原来的.bmp文件
                    os.remove(file_path)
                    print(f'Deleted original file {file_path}')
            except IOError:
                print(f'Failed to process {file_path}. It may not be a valid image file.')

# 使用函数
convert_bmp_to_png_and_delete('/media/jnu/1eb5aebb-d8ad-4c4a-a83d-4a86dd7d859a/1000张/sp')