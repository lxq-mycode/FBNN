from PIL import Image
import numpy as np
import cv2
import numpy as np
from PIL import Image
test_matfocus_noise_6464_path_plus = '/media/jnu/data2/model_12_9/focus/LQ_1'
test_noise = []
test_clear = []

for idx in range(1, 10001):
    # get the match imgs of clears and noises
    n_path = ('%s/%d.png' % (test_matfocus_noise_6464_path_plus, idx))
    img = cv2.imread(n_path, cv2.IMREAD_GRAYSCALE)
    # append clear imgs and noise imgs
    # 归一    # 将图像转换为NumPy数组
    image_array = np.array(img)

    # 判断像素值是否大于0
    mask = image_array > 100

    # 将大于0的像素值设置为255
    image_array[mask] = 250

    # 将NumPy数组转换为图像并保存
    result_image = Image.fromarray(image_array)
    result_image.save(test_matfocus_noise_6464_path_plus+'/%d.png'% (idx))




