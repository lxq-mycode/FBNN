import os
from PIL import Image

# 定义原始文件夹路径和目标文件夹路径
# original_folder_path = '/media/jnu/data2/minist/'
# target_folder_path = '/media/jnu/data1/focus_test/net/LQ/'

# original_folder_path = '/home/jnu/data/datas/ampSLM_YH_squared_train6464/'
# target_folder_path = '/home/jnu/data/datas/data_cut/YH_train6464_tm_to64_1/'
# original_folder_path = '/media/jnu/data1/focus_test/net_1/LQ'
# target_folder_path = '/media/jnu/data1/focus_test/net_1/LQ'

original_folder_path = "/media/jnu/data2/xwx/GS/80/face/our_face_HQ"
target_folder_path = "/media/jnu/data2/xwx/GS/80/face/our_face_HQ"


# 创建目标文件夹
os.makedirs(target_folder_path, exist_ok=True)

# 遍历原始文件夹中的所有图片文件
for file in os.listdir(original_folder_path):
    file_path = os.path.join(original_folder_path, file)

    # 确保当前路径是一个文件
    if os.path.isfile(file_path):
        # 打开图片文件
        img = Image.open(file_path)

        # 裁剪图片为128x128大小
        img = img.resize((80,80),Image.BILINEAR)

        # 保存裁剪后的图片到目标文件夹
        target_file_path = os.path.join(target_folder_path, file)
        img.save(target_file_path)

print("图片裁剪完成！")


