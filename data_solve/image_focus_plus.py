from PIL import Image, ImageDraw, ImageFont

# 创建一个函数来绘制点并保存图像
def draw_dot_and_save(dot_position, output_path):
    # 创建空白图像
    width, height = 256, 256
    image = Image.new('L', (width, height), color=0)

    # 获取绘图对象
    draw = ImageDraw.Draw(image)

    # 设置字体大小
    font_size = 100  # 使用较小的字体大小来绘制点
    font_path = "/home/jnu/下载/Roboto/Roboto-Bold.ttf"
    # 加载字体，并指定大小
    font = ImageFont.truetype(font_path, font_size)

    # 计算文本位置
    text = '.'
    text_width, text_height = draw.textsize(text, font=font)

    x, y = dot_position
    x = x - text_width // 2
    y = y - text_height // 2

    # 绘制文本
    draw.text((x, y), text, font=font, fill=255)

    # 保存图像
    image.save(output_path)

# 中间点
middle_position = (256 // 2, 256 // 2-30)
draw_dot_and_save(middle_position, '/home/jnu/下载/1.png')

# 上面点
top_position = (256 // 2, 256 // 4-40)
draw_dot_and_save(top_position, '/home/jnu/下载/2.png')

# 下面点
bottom_position = (256 // 2, 3 * 256 // 4-20)
draw_dot_and_save(bottom_position, '/home/jnu/下载/3.png')

# 左面点
left_position = (256 // 4, 256 // 2-40)
draw_dot_and_save(left_position, '/home/jnu/下载/4.png')

# 右面点
right_position = (3 * 256 // 4, 256 // 2-40)
draw_dot_and_save(right_position, '/home/jnu/下载/5.png')


from PIL import Image, ImageDraw, ImageFont

# 创建一个函数来绘制字母并保存图像
# def draw_letter_and_save(letter, output_path):
#     # 创建空白图像
#     width, height = 256, 256
#     image = Image.new('L', (width, height), color=0)
#
#     # 获取绘图对象
#     draw = ImageDraw.Draw(image)
#
#     # 设置字体大小
#     font_size = 200  # 使用较大的字体大小来绘制字母
#     font_path = "/home/jnu/下载/Roboto/Roboto-Bold.ttf"
#     # 加载字体，并指定大小
#     font = ImageFont.truetype(font_path, font_size)
#
#     # 计算文本位置
#     text_width, text_height = draw.textsize(letter, font=font)
#
#     x = (width - text_width) // 2
#     y = (height - text_height) // 2
#
#     # 绘制文本
#     draw.text((x, y), letter, font=font, fill=255)
#
#     # 保存图像
#     image.save(output_path)
#
# # 生成包含不同字母的图片
# letters = ['1', '2', '3', '4', '5']
# for i, letter in enumerate(letters):
#     output_path = f'/home/jnu/下载/{i+1}.png'
#     draw_letter_and_save(letter, output_path)