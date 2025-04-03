# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image,ImageDraw
# import cv2
#
#
# image_array0 = Image.new('L',(64,64))
# image_array1 = Image.new('L',(64,64))
# image_array2 = Image.new('L',(64,64))
# image_array3 = Image.new('L',(64,64))
# image_array4 = Image.new('L',(64,64))
# # image_array0[28:36, 28:36] = 255
# # image_array1[28:36, 8:16] = 255
# # image_array2[28:36, 48:56] = 255
# # image_array3[8:16, 28:36] = 255
# # image_array4[48:56, 28:36] = 255
# # 定义三角形三个顶点坐标
# triangle_points = [(32, 28), (36, 28), (34, 32)]
# # 定义矩形的左上角和右下角坐标
# rectangle_left_top = (28, 31)
# rectangle_right_bottom = (36, 33)
# # 定义菱形的四个顶点坐标
# diamond_points = [(32, 29), (35, 32), (32, 35), (29, 32)]
# # 定义正方形的左上角和右下角坐标
# square_left_top = (31, 31)
# square_right_bottom = (34, 34)
# # # 定义梯形的四个顶点坐标
# # trapezoid_points = [(29, 30), (34, 30), (33, 34), (30, 34)]
# # 定义矩形的左上角和右下角坐标
# # 定义字母E的坐标点
# e_coordinates = [(10, 10), (18, 10), (10, 30), (18, 30), (10, 20), (16, 20
# )]
#
#
#
#
# # 在图像中绘制矩形
#
#
# draw = ImageDraw.Draw(image_array0)
# draw1 = ImageDraw.Draw(image_array1)
# draw2 = ImageDraw.Draw(image_array2)
# draw3 = ImageDraw.Draw(image_array3)
# draw4 = ImageDraw.Draw(image_array4)
#
#
#
# #三角形
# draw.polygon(triangle_points,fill=255)
#
# # 在图像中绘制矩形
# draw1.rectangle([rectangle_left_top, rectangle_right_bottom], fill=255)
# # 在图像中绘制菱形
# draw2.polygon(diamond_points, fill=255)
# # 在图像中绘制正方形
# draw3.rectangle([square_left_top, square_right_bottom], fill=255)
# # 在图像中绘制梯形
# # draw4.rectangle([rectangle_left_top1, rectangle_right_bottom1], fill=255)
# # 在图像中绘制字母E
# draw4.polygon(e_coordinates, fill=255)
#
# image_array0.save("/media/jnu/data2/foucs_image_plus/1.png")
# image_array1.save("/media/jnu/data2/foucs_image_plus/2.png")
# image_array2.save("/media/jnu/data2/foucs_image_plus/3.png")
# image_array3.save("/media/jnu/data2/foucs_image_plus/4.png")
# image_array4.save("/media/jnu/data2/foucs_image_plus/5.png")
#
#


from PIL import Image, ImageDraw

# 创建一个64x64的全黑图像
image_size = (256, 256)
image = Image.new("L", image_size, color=0)
image1 = Image.new("L", image_size, color=0)
image2 = Image.new("L", image_size, color=0)
image3 = Image.new("L", image_size, color=0)
image4 = Image.new("L", image_size, color=0)

# 获取图像的绘制对象
draw = ImageDraw.Draw(image)
draw1 = ImageDraw.Draw(image1)
draw2 = ImageDraw.Draw(image2)
draw3 = ImageDraw.Draw(image3)
draw4 = ImageDraw.Draw(image4)
# 定义要绘制的点的坐标
points = [(32, 29), (33, 29), (34, 29),(32,30), (32, 31),(32,32),(33,32),(32,34),(34,32),(32,35),(33,35),(34,35),(32,33),(35,29),(35,32),(35,35)]
points1 = [(32, 29), (32, 30), (32, 31),(32,32), (32, 33),(32,34),(32,35),(33,31),(34,30),(35,29),(33,33),(34,34),(35,35)]
points2 = [(32, 29), (32, 30), (32, 31),(32,32), (32, 33),(32,34),(32,35),(30,29),(31,29),(33,29),(34,29),(30,35),(31,35),(33,35),(34,35)]
points3 = [(29, 29), (30, 30),(31,31), (33, 33),(32,32),(34,34),(35,35),(33,31),(34,30),(35,29),(31,33),(30,34),(29,35),(31,32),(33,32)]
points4 = [(30, 33), (30, 34),(30, 35), (30, 28), (30, 29), (30, 30),(30,31), (30,32),(31, 32),(32,32),(33,32)
    ,(34,32),(34,33),(34,35),(34,34),(34,28),(34,29),(34,30),(34,31),(34,34)]
points = [(x + 96, y + 96) for x, y in points]
points1 = [(x + 96, y + 96) for x, y in points1]
points2 = [(x + 96, y + 96) for x, y in points2]
points3= [(x + 96, y + 96) for x, y in points3]
points4 = [(x + 96, y + 96) for x, y in points4]

# 在图像中绘制点
for point in points:
    draw.point(point, fill=255)
# 在图像中绘制点
for point in points1:
    draw1.point(point, fill=255)
# 在图像中绘制点
for point in points2:
    draw2.point(point, fill=255)
# 在图像中绘制点
for point in points3:
    draw3.point(point, fill=255)
# 在图像中绘制点
for point in points4:
    draw4.point(point, fill=255)
# 保存图像
image.save("/media/jnu/data1/focus_test/net/GT/5.png")
image1.save("/media/jnu/data1/focus_test/net/GT//4.png")
image2.save("/media/jnu/data1/focus_test/net/GT//3.png")
image3.save("/media/jnu/data1/focus_test/net/GT//2.png")
image4.save("/media/jnu/data1/focus_test/net/GT//1.png")
# 显示图像（可选）
image4.show()
