from PIL import Image, ImageDraw, ImageFont
import random
# 设置字体和大小
font = ImageFont.truetype("/home/jnu/下载/Noto_Sans_SC/static/3.ttf", 64)  # 确保simhei.ttf文件在当前目录中

# 要写入的字母或汉字列表
characters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
# characters = ["一", "乙", "十", "厂", "人", "太", "天", "木", "犬", "末", "电", "出", "世", "东", "西", "车", "果","费","餐","嬴"]

# 像素值范围
min_val, max_val = 0, 15

for i, char in enumerate(characters):
    # 创建一个256x256的黑色背景图像
    image = Image.new("L", (256, 256))

    # 为每个像素点随机分配一个像素值
    pixels = image.load()
    for x in range(256):
        for y in range(256):
            # 随机生成一个像素值
            pixel_value = random.randint(min_val, max_val)
            # 将这个值赋给当前像素
            pixels[x, y] = pixel_value

    draw = ImageDraw.Draw(image)

    # 计算字母或汉字的宽度和高度
    text_width, text_height = draw.textsize(char, font=font)

    # 计算字母或汉字放置的位置，使其居中
    x = (256 - text_width) // 2
    y = (256 - text_height) // 2

    # 在图像上绘制白色的字母或汉字
    draw.text((x, y), char, font=font, fill="white")

    # 保存图像，文件名为 1.png, 2.png, 等
    image.save(f'/media/jnu/data2/model18jujiao_weitiao/LQ/{i+1}.png')

    # 如果你想要显示生成的每个图像，可以取消下面的注释
    # image.show()
