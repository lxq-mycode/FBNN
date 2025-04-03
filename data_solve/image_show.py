import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
# def f(x):
#     return 1/ (1+np.exp(-20*(x-0.5)))
#     # return (np.exp(5*x) - np.exp(-5*x)) / (np.exp(5*x) + np.exp(-5*x))
#  # 1/ (1+torch.exp(-self.alpha*(x-0.5)))
# x = np.linspace(0, 1, 100)  # 生成 x 值的数组
# y = f(x)                    # 计算对应的 y 值
#
# plt.plot(x, y)              # 绘制曲线
# plt.xlabel('x')             # 设置 x 轴标签
# plt.ylabel('y')             # 设置 y 轴标签
# plt.title('Graph of f(x) = np.exp(10*x) - np.exp(-10*x)) / (np.exp(10*x) + np.exp(-10*x)')  # 设置标题
# plt.grid(True)              # 显示网格
# plt.show()                  # 显示图像
#
#
#


image_array0 = np.zeros((64,64),dtype=np.uint8)
image_array1 = np.zeros((64,64),dtype=np.uint8)
image_array2 = np.zeros((64,64),dtype=np.uint8)
image_array3 = np.zeros((64,64),dtype=np.uint8)
image_array4 = np.zeros((64,64),dtype=np.uint8)
#
# image_array0[28:36, 28:36] = 255
# image_array1[28:36, 8:16] = 255
# image_array2[28:36, 48:56] = 255
# image_array3[8:16, 28:36] = 255
# image_array4[48:56, 28:36] = 255
radius = 2
center_x0 = 32
center_y0 = 32
center_x1 = 32
center_y1 = 12
center_x2 = 32
center_y2 = 52
center_x3 = 12
center_y3 = 32
center_x4 = 52
center_y4 = 32

cv2.circle(image_array0,(center_x0,center_y0),radius,255,-1)
cv2.circle(image_array1,(center_x1,center_y1),radius,255,-1)
cv2.circle(image_array2,(center_x2,center_y2),radius,255,-1)
cv2.circle(image_array3,(center_x3,center_y3),radius,255,-1)
cv2.circle(image_array4,(center_x4,center_y4),radius,255,-1)




image_array0 = Image.fromarray(image_array0)
image_array1 = Image.fromarray(image_array1)
image_array2 = Image.fromarray(image_array2)
image_array3 = Image.fromarray(image_array3)
image_array4 = Image.fromarray(image_array4)

image_array0.save("/media/jnu/data2/foucs_image_plus/1.png")
image_array1.save("/media/jnu/data2/foucs_image_plus/2.png")
image_array2.save("/media/jnu/data2/foucs_image_plus/3.png")
image_array3.save("/media/jnu/data2/foucs_image_plus/4.png")
image_array4.save("/media/jnu/data2/foucs_image_plus/5.png")


