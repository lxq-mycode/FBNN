from PIL import Image, ImageDraw, ImageFont
import random
# 设置字体和大小
font = ImageFont.truetype("/home/jnu/下载/Noto_Sans_SC/static/2.ttf", 35)  # 确保simhei.ttf文件在当前目录中


# 从Unicode范围中选取常用汉字的范围
# 常用汉字范围：基本汉字 (4E00 - 9FFF)
# def generate_unique_characters(num):
#     characters = set()
#
#     # 使用循环确保生成的字符是唯一的
#     while len(characters) < num:
#         # 从4E00 (19968) 到 9FFF (40959) 范围内选择汉字
#         char = chr(random.randint(0x4E00, 0x9FFF))
#         characters.add(char)
#
#     return list(characters)


# 生成2000个不重复的汉字
# unique_characters = generate_unique_characters(2000)
# characters = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E",
#               "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U",
#               "V", "W", "X", "Y", "Z", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k",
#               "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "中", "国"]

# 打印生成的字符数量和前10个字符以作检查
# print(f"生成了 {len(unique_characters)} 个不重复的字符")
# print(unique_characters[:10])  # 打印前10个字符检查

# 如果要将这些字符用于图像生成，可以将列表传递给生成图像的代码

# # 要写入的汉字列表
# unique_characters=unique_characters + characters
# characters =["A", "B", "C", "D", "E",
#               "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U",
#               "V", "W", "X", "Y", "Z",]
characters =["0", "5", "E", "字", "体"]
# characters = ["一", "乙", "十", "厂", "人", "太", "天", "木", "犬", "末", "电", "出", "世", "东", "西", "车", "果","费","餐","嬴"]
# 生成图像
# 像素值范围
min_val, max_val = 0,10

# 每个点的大小
block_size = 8
for i, char in enumerate(characters):
    # 创建一个256x256的黑色背景图像
    # image = Image.new("RGB", (256, 256), "black")
    image = Image.new("L", (256, 256))
    # 计算总共有多少个点
    # num_blocks = (256 // block_size) * (256 // block_size)
    #
    # # 为每个8x8的点随机分配一个像素值
    # pixels = image.load()
    # for k in range(0, 256, block_size):
    #     for p in range(0, 256, block_size):
    #         # 随机生成一个像素值
    #         pixel_value = random.randint(min_val, max_val)
    #         # 将这个值赋给当前8x8区域的所有像素
    #         for x in range(k, k + block_size):
    #             for y in range(p, p + block_size):
    #                 pixels[x, y] = pixel_value

    draw = ImageDraw.Draw(image)

    # 计算汉字的宽度和高度
    text_width, text_height = draw.textsize(char, font=font)

    # 计算汉字放置的位置，使其居中
    x = (256 - text_width) // 2
    y = (256 - text_height) // 2

    # 在图像上绘制白色的汉字
    draw.text((x, y), char, font=font, fill="white")

    # 保存图像，文件名为 1.png, 2.png, 等
    image.save(f'/media/jnu/data2/model18jujiao_weitiao/LQ_1/{i+1}.png')

    # 如果你想要显示生成的每个图像，可以取消下面的注释
