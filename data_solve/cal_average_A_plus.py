import os
import cv2
import numpy as np

# 定义输入文件夹路径
folder1 = '/media/jnu/SUCCESS/data_7_5/hanzi_1/duli_net/yuce/testmodel_clear1'
folder2 = '/media/jnu/SUCCESS/data_7_5/hanzi_1/prvamp/LQ/'

# 确保两个文件夹中都有相同数量的图片
files1 = sorted(os.listdir(folder1))
files2 = sorted(os.listdir(folder2))

if len(files1) != len(files2):
    print("Folders do not contain the same number of images.")
    exit()

# 遍历每对图片进行计算
for file1, file2 in zip(files1, files2):
    if file1.endswith('.png') and file2.endswith('.png'):
        # 构建图片路径
        path1 = os.path.join(folder1, file1)
        path2 = os.path.join(folder2, file2)

        # 读取图片
        image1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
        image2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)

        # 检查图片是否读取成功
        if image1 is None:
            print(f"Failed to load image: {path1}")
            continue

        if image2 is None:
            print(f"Failed to load image: {path2}")
            continue

        # 确保两张图片的尺寸相同
        if image1.shape != image2.shape:
            print(f"Images {file1} and {file2} do not have the same dimensions.")
            continue

        # 找到 image2 中像素值为 255 的位置
        mask = image2 == 255

        # 在 image1 中找到这些位置的对应像素值
        corresponding_pixels = image1[mask]

        # 计算这些像素值的平均值
        mean_value = np.mean(corresponding_pixels)
        # mean_value = mean_value/2.64088
        print(f"The mean value of corresponding pixels for {file1} is: {mean_value}")

'''
点的聚焦
整个图片的平均强度
net 33.928955078125      4.73
prvamp 29.231979370117188    4.76
duli_net 21.007186889648438   



dulinet 
独立net
The mean value of corresponding pixels for 1.png is: 93.984375
The mean value of corresponding pixels for 2.png is: 159.15625
The mean value of corresponding pixels for 3.png is: 142.84375
The mean value of corresponding pixels for 4.png is: 133.234375
The mean value of corresponding pixels for 5.png is: 163.5625

net  
The mean value of corresponding pixels for 1.png is: 160.390625
The mean value of corresponding pixels for 2.png is: 201.921875
The mean value of corresponding pixels for 3.png is: 179.484375
The mean value of corresponding pixels for 4.png is: 169.828125
The mean value of corresponding pixels for 5.png is: 203.4375


prvamp
The mean value of corresponding pixels for 1.png is: 139.125
The mean value of corresponding pixels for 2.png is: 145.328125
The mean value of corresponding pixels for 3.png is: 154.1875
The mean value of corresponding pixels for 4.png is: 130.5625
The mean value of corresponding pixels for 5.png is: 148.53125


区域
prvamp
The mean value of corresponding pixels for 1.png is: 165.55555555555554
The mean value of corresponding pixels for 2.png is: 117.32330827067669
The mean value of corresponding pixels for 3.png is: 124.63225806451612
The mean value of corresponding pixels for 4.png is: 121.62348178137651
The mean value of corresponding pixels for 5.png is: 123.22591362126246
prvamp
第 1 张图片的平均强度是 36.3046875
第 2 张图片的平均强度是 27.598876953125
第 3 张图片的平均强度是 29.106613159179688
第 4 张图片的平均强度是 34.12541198 730469
第 5 张图片的平均强度是 33.25926208496094

net
The mean value of corresponding pixels for 1.png is: 171.37254901960785
The mean value of corresponding pixels for 2.png is: 159.11278195488723
The mean value of corresponding pixels for 3.png is: 170.90322580645162
The mean value of corresponding pixels for 4.png is: 123.74898785425101
The mean value of corresponding pixels for 5.png is: 139.3920265780731
第 1 张图片的平均强度是 22.237060546875
第 2 张图片的平均强度是 21.561492919921875
第 3 张图片的平均强度是 20.975112915039062
第 4 张图片的平均强度是 20.045303344726562
第 5 张图片的平均强度是 21.951187133789062


duli_net
The mean value of corresponding pixels for 1.png is: 120.0
The mean value of corresponding pixels for 2.png is: 112.86466165413533
The mean value of corresponding pixels for 3.png is: 134.4967741935484
The mean value of corresponding pixels for 4.png is: 86.1497975708502
The mean value of corresponding pixels for 5.png is: 90.70099667774086
第 1 张图片的平均强度是 21.2972412109375
第 2 张图片的平均强度是 20.905471801757812
第 3 张图片的平均强度是 20.664886474609375
第 4 张图片的平均强度是 24.405609130859375
第 5 张图片的平均强度是 22.9661865234375



'''












# 点 的
# net 的 yita
#The mean value of corresponding pixels for 1.png is: 60.73378002786949  中
# The mean value of corresponding pixels for 2.png is: 76.46007202144739  右
# The mean value of corresponding pixels for 3.png is: 67.96385106479659  下
# The mean value of corresponding pixels for 4.png is: 64.30739942746357   上
# The mean value of corresponding pixels for 5.png is: 77.03398109721002    左


# dulinet 的 yita
# The mean value of corresponding pixels for 1.png is: 35.58827928569264
# The mean value of corresponding pixels for 2.png is: 60.2663695434855
# The mean value of corresponding pixels for 3.png is: 54.089451243524884
# The mean value of corresponding pixels for 4.png is: 50.450749371421644
# The mean value of corresponding pixels for 5.png is: 61.93484747508406

# prvamp 的 yita
# The mean value of corresponding pixels for 1.png is: 52.68130320196298
# The mean value of corresponding pixels for 2.png is: 55.030188800702796
# The mean value of corresponding pixels for 3.png is: 58.38489442912968
# The mean value of corresponding pixels for 4.png is: 49.43901275332465
# The mean value of corresponding pixels for 5.png is: 56.24308942473721


# 区域 的
# net 的 yita
# The mean value of corresponding pixels for 1.png is: 44.63953725341734
# The mean value of corresponding pixels for 2.png is: 53.164096816212776
# The mean value of corresponding pixels for 3.png is: 65.41380146011934
# The mean value of corresponding pixels for 4.png is: 46.32982761485381
# The mean value of corresponding pixels for 5.png is: 60.784011994657966



# dulinet 的 yita
# The mean value of corresponding pixels for 1.png is: 26.703666443458516
# The mean value of corresponding pixels for 2.png is: 32.68920241342477
# The mean value of corresponding pixels for 3.png is: 42.31544030777619
# The mean value of corresponding pixels for 4.png is: 28.867787872362104
# The mean value of corresponding pixels for 5.png is: 42.60405437655297

# prvamp 的 yita
# The mean value of corresponding pixels for 1.png is: 37.96588703369316
# The mean value of corresponding pixels for 2.png is: 31.64376696568864
# The mean value of corresponding pixels for 3.png is: 43.085035066874454
# The mean value of corresponding pixels for 4.png is: 37.16009570118806
# The mean value of corresponding pixels for 5.png is: 39.42279268463962