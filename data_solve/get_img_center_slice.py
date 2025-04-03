# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
#
# def read_image(path):
#     """读取图片并转换为灰度图"""
#     image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#     if image is None:
#         raise ValueError(f"Image at {path} could not be read")
#     return image
#
# def get_center_slice(image):
#     """提取图片的中心截面"""
#     center_index = image.shape[0] // 2
#     return image[center_index, :]
#
# # 图片路径列表
# image_paths = [
#     ('/media/jnu/SUCCESS/data_5_29/jujiao/dian/true/1.png','original'),
#     ('/media/jnu/SUCCESS/data_5_29/jujiao/dian/Prvamp_20000/jujiao/1.png','Prvamp'),
#     ('/media/jnu/SUCCESS/data_5_29/jujiao/dian/IT_20000_jujiao/yuce/testmodel_clear1/1.png','Independent Training-20000'),
#     ('/media/jnu/SUCCESS/data_5_29/jujiao/dian/IT_2000_jujiao/yuce/testmodel_clear1/1.png','Independent Training-2000'),
#     ('/media/jnu/SUCCESS/data_5_29/jujiao/dian/2000_20000_jujiao/jujiaoyuce/testmodel_clear1/1.png','Joint Training-20000'),
#     ('/media/jnu/SUCCESS/data_5_29/jujiao/dian/2000_2000_jujiao/jujiaoyuce/1.png','Joint Training-2000'),
# ]
#
# # 读取并调整所有图片的大小为相同
# images = [read_image(path) for path, method in image_paths]
#
# # 获取每张图片的中心截面
# center_slices = [get_center_slice(img)/255.0 for img in images]
# x_values=np.arange(-128,128)
# # 绘制强度图
# plt.figure(figsize=(10, 6))
# for i, (center_slice, (_, method)) in enumerate(zip(center_slices, image_paths)):
#     plt.plot(x_values, center_slice, label=method)
#
# plt.xlabel('Pixel Index')
# plt.ylabel('Intensity')
# plt.title('Center Slice Intensity Comparison')
# plt.legend()
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the image
image_path = '/media/jnu/data2/10_25shiyanzhengli/jujiao/dot/yuce/testmodel_clear1/1.png'
image = Image.open(image_path).convert('L')  # Convert to grayscale
image_array = np.array(image)

# Extract the rows from 124 to 132
rows_to_extract = np.arange(124, 132)
extracted_pixels = image_array[rows_to_extract, :]

# Calculate the average pixel values across the selected rows
average_pixel_values = np.mean(extracted_pixels, axis=0)

# Normalize the pixel values to range [0, 1]
normalized_pixel_values = average_pixel_values / 255.0

# Plot the normalized intensity distribution
plt.figure(figsize=(10, 4))
plt.plot(normalized_pixel_values)
plt.title('Normalized Intensity Distribution (Rows 124 to 132)')
plt.xlabel('Pixel Position (0 to 255)')
plt.ylabel('Normalized Intensity (0 to 1)')
plt.xlim(0, 255)
plt.ylim(0, 1)
plt.grid(True)
plt.show()