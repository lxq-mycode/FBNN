from PIL import Image, ImageDraw

# 创建一个函数来绘制点并保存图像
def draw_dot_and_save(dot_position, output_path):
    # 创建空白图像
    width, height = 256, 256
    image = Image.new('L', (width, height), color=0)

    # 获取绘图对象
    draw = ImageDraw.Draw(image)

    # 计算8x8正方形的位置
    x, y = dot_position
    x0, y0 = x, y
    x1, y1 = x + 8, y + 8

    # 绘制8x8正方形
    draw.rectangle([x0, y0, x1, y1], fill=255)

    # 保存图像
    image.save(output_path)

# 生成1024张图片
count = 0
for i in range(0, 256, 8):
    for j in range(0, 256, 8):
        dot_position = (i, j)
        output_path = f'/media/jnu/data2/10_29/focus_all/LQ/{count + 1}.png'
        draw_dot_and_save(dot_position, output_path)
        count += 1

print("图片生成完成")


