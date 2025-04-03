# from PIL import Image, ImageDraw
#
#
# def create_image_with_shape(shape_func, filename):
#     # 创建一个256x256的全黑图片
#     img = Image.new('L', (256, 256), 0)
#     draw = ImageDraw.Draw(img)
#
#     # 中间8x8区域的左上角坐标
#     start_x, start_y = 124, 124
#
#     # 调用传入的形状绘制函数
#     shape_func(draw, start_x, start_y)
#
#     # 保存图片
#     img.save(filename)
#
#
# def draw_star(draw, x, y):
#     # 五角星的顶点坐标
#     points = [
#         (x + 4, y), (x + 5, y + 3), (x + 8, y + 3),
#         (x + 6, y + 5), (x + 7, y + 8), (x + 4, y + 6),
#         (x + 1, y + 8), (x + 2, y + 5), (x, y + 3),
#         (x + 3, y + 3)
#     ]
#     draw.polygon(points, fill=255)
#
#
# def draw_triangle(draw, x, y):
#     # 三角形的顶点坐标
#     points = [(x + 4, y), (x, y + 8), (x + 8, y + 8)]
#     draw.polygon(points, fill=255)
#
#
# def draw_trapezoid(draw, x, y):
#     # 梯形的顶点坐标
#     points = [(x + 2, y), (x + 6, y), (x + 8, y + 8), (x, y + 8)]
#     draw.polygon(points, fill=255)
#
#
# def draw_parallelogram(draw, x, y):
#     # 平行四边形的顶点坐标
#     points = [(x + 2, y), (x + 8, y), (x + 6, y + 8), (x, y + 8)]
#     draw.polygon(points, fill=255)
#
#
# def draw_pentagon(draw, x, y):
#     # 五边形的顶点坐标
#     points = [(x + 4, y), (x + 7, y + 3), (x + 6, y + 8), (x + 2, y + 8), (x + 1, y + 3)]
#     draw.polygon(points, fill=255)
#
#
# # 创建并保存五张图片
# create_image_with_shape(draw_star, '/media/jnu/data2/data_8_11/focus_patten/LQ/1.png')
# create_image_with_shape(draw_triangle, '/media/jnu/data2/data_8_11/focus_patten/LQ/2.png')
# create_image_with_shape(draw_trapezoid, '/media/jnu/data2/data_8_11/focus_patten/LQ/3.png')
# create_image_with_shape(draw_parallelogram, '/media/jnu/data2/data_8_11/focus_patten/LQ/4.png')
# create_image_with_shape(draw_pentagon, '/media/jnu/data2/data_8_11/focus_patten/LQ/5.png')
#
from PIL import Image, ImageDraw


def create_image_with_shape(shape_func, filename):
    # 创建一个全黑的256x256图片
    img = Image.new('L', (256, 256), 0)
    draw = ImageDraw.Draw(img)

    # 中间16x16区域的左上角坐标
    start_x, start_y = 120, 120

    # 调用传入的形状绘制函数
    shape_func(draw, start_x, start_y)

    # 保存图片
    img.save(filename)


def draw_star(draw, start_x, start_y):
    # 五角星的顶点坐标
    points = [
        (start_x + 8, start_y), (start_x + 10, start_y + 6),
        (start_x + 16, start_y + 6), (start_x + 11, start_y + 10),
        (start_x + 13, start_y + 16), (start_x + 8, start_y + 12),
        (start_x + 3, start_y + 16), (start_x + 5, start_y + 10),
        (start_x, start_y + 6), (start_x + 6, start_y + 6)
    ]
    draw.polygon(points, fill=255)


def draw_triangle(draw, start_x, start_y):
    # 三角形的顶点坐标
    points = [
        (start_x + 8, start_y), (start_x, start_y + 16), (start_x + 16, start_y + 16)
    ]
    draw.polygon(points, fill=255)


def draw_trapezoid(draw, start_x, start_y):
    # 梯形的顶点坐标
    points = [
        (start_x + 4, start_y), (start_x + 12, start_y),
        (start_x + 16, start_y + 16), (start_x, start_y + 16)
    ]
    draw.polygon(points, fill=255)


def draw_parallelogram(draw, start_x, start_y):
    # 平行四边形的顶点坐标
    points = [
        (start_x + 4, start_y), (start_x + 12, start_y),
        (start_x + 8, start_y + 16), (start_x, start_y + 16)
    ]
    draw.polygon(points, fill=255)


def draw_pentagon(draw, start_x, start_y):
    # 五边形的顶点坐标
    points = [
        (start_x + 8, start_y), (start_x + 14, start_y + 6),
        (start_x + 11, start_y + 16), (start_x + 5, start_y + 16),
        (start_x + 2, start_y + 6)
    ]
    draw.polygon(points, fill=255)


# 创建并保存五张图片
# create_image_with_shape(draw_star, '1.png')
# create_image_with_shape(draw_triangle, '2.png')
# create_image_with_shape(draw_trapezoid, '3.png')
# create_image_with_shape(draw_parallelogram, '4.png')
# create_image_with_shape(draw_pentagon, '5.png')

create_image_with_shape(draw_star, '/media/jnu/data2/data_8_11/focus_patten16*16/LQ/1.png')
create_image_with_shape(draw_triangle, '/media/jnu/data2/data_8_11/focus_patten16*16/LQ/2.png')
create_image_with_shape(draw_trapezoid, '/media/jnu/data2/data_8_11/focus_patten16*16/LQ/3.png')
create_image_with_shape(draw_parallelogram, '/media/jnu/data2/data_8_11/focus_patten16*16/LQ/4.png')
create_image_with_shape(draw_pentagon, '/media/jnu/data2/data_8_11/focus_patten16*16/LQ/5.png')