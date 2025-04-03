import cv2
import os
from natsort import natsorted

# 图片文件夹路径
image_folder_path = '/media/jnu/data2/shiyan_9_8/test/result/yuce/testmodel_clear1'

# 视频保存文件夹路径
video_save_folder_path = '/media/jnu/data2/shiyan_9_8/test/result/yuce'
video_save_path = os.path.join(video_save_folder_path, 'output_video_net.mp4')

# 如果保存文件夹不存在，创建文件夹
if not os.path.exists(video_save_folder_path):
    os.makedirs(video_save_folder_path)

# 获取所有的.png文件并排序
images = [img for img in os.listdir(image_folder_path) if img.endswith('.png')]
images = natsorted(images)

# 检查图片是否为空
if not images:
    raise ValueError("No .png images found in the folder")

# 读取第一张图片以获取视频的宽度和高度
frame = cv2.imread(os.path.join(image_folder_path, images[0]))
height, width, layers = frame.shape

# 计算每秒显示的帧数（即fps）
total_images = len(images)
total_seconds = 62
fps = total_images / total_seconds

# 定义视频编解码器和创建 VideoWriter 对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用'mp4v'编解码器
video = cv2.VideoWriter(video_save_path, fourcc, fps, (width, height))

for image in images:
    img_path = os.path.join(image_folder_path, image)
    frame = cv2.imread(img_path)
    video.write(frame)

# 释放 VideoWriter
video.release()

print("Video created successfully!")