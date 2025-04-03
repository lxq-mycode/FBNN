import cv2
import numpy as np
import math
from skimage.metrics import structural_similarity as ssim
from skimage import io
# 计算5张图片的PSNR
psnr_sum = 0
# test_clear_path='/media/jnu/data1/TM_test/net_1000_dot/LQ'
# test_clear_path='/media/jnu/630C-76C8/data1/true'
# test_clear_path='/media/jnu/data1/TM_test/net/testnoise1'
# test_clear_path='/media/jnu/data1/TM_test/net_500_dot/LQ'
# test_clear_path = '/media/jnu/SUCCESS/data_5_29/chengxiang/true'


test_noise_2000_path =  '/media/jnu/data1/TM_test/net_2000_dot/testclear1'
test_noise_5000_path = '/media/jnu/data1/TM_test/net_5000_dot/testclear1'
test_noise_1000_path = '/media/jnu/data1/TM_test/net_1000_dot/testclear1'
test_noise_500_path = '/media/jnu/data1/TM_test/net_500_dot/testclear1'


# test_noise = '/media/jnu/630C-76C8/data1/for5000/'/media/jnu/data1/now_shiyan/2000_duibi/yuce/testmodel_clear1
# test_noise = '/media/jnu/630C-76C8/data/gs64/'
# test_noise = '/media/jnu/630C-76C8/data/MAP64/'
# test_noise = '/media/jnu/SUCCESS/data_6_28/jujiao/quyu/prvamp/yuce'
test_clear_path = '/home/jnu/音乐/NWDN_data_true/yuce'
test_noise = '/home/jnu/音乐/NWDN_data_true/yucerefusion'


# 计算每张图片的PSNR并输出结果
psnrs = []

def psnr1(img1, img2):
    mse = np.mean((clear_data / 1.0 - noise_data / 1.0) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)


def psnr2(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


for idx in range(1, 16):
    c_path = ('%s/%d.png' % (test_clear_path, idx))
    n_path = ('%s/a%d.png' % (test_noise, idx))
    # append clear imgs and noise imgs
    clear_data = io.imread(c_path, cv2.IMREAD_GRAYSCALE)
    noise_data = io.imread(n_path, cv2.IMREAD_GRAYSCALE)
    # 加载压缩后的图像
    # 计算MSE
    # 计算PSNR
    psnr = psnr2(clear_data,noise_data)

    print('image{}.png psnr: {:.2f} dB'.format(idx, psnr))
    psnrs.append(psnr)

# 计算平均PSNR并输出结果
mean_psnr = np.mean(psnrs)
print('平均psnr: {:.2f} '.format(mean_psnr))
