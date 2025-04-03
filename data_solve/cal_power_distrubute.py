import cv2
import numpy as np

# 计算5张图片的PSNR
psnr_sum = 0
test_clear_path='/media/jnu/data1/TM_test/net_1000_dot/LQ'
test_noise_2000_path =  '/media/jnu/data1/TM_test/net_2000_dot/testclear1'
test_noise_5000_path = '/media/jnu/data1/TM_test/net_5000_dot/testclear1'
test_noise_1000_path = '/media/jnu/data1/TM_test/net_1000_dot/testclear1'
test_noise_500_path = '/media/jnu/data1/TM_test/net_500_dot/testclear1'


# 计算每张图片的PSNR并输出结果
psnrs = []
count=0
center_x=0
center_y=0

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
def dot_set(idx):
   if idx==1:
       center_x = 128
       center_y = 128
   elif idx==2:
       center_x = 128
       center_y = 48
   elif idx==3:
       center_x = 128
       center_y = 128
   elif idx == 4:
        center_x = 128
        center_y = 128
   elif idx == 5:
       center_x = 128
       center_y = 128

   return


for idx in range(1, 6):

    c_path = ('%s/%d.png' % (test_clear_path, idx))
    n_path = ('%s/%d.png' % (test_noise_500_path, idx))
    # append clear imgs and noise imgs
    clear_data = cv2.imread(c_path, cv2.IMREAD_GRAYSCALE)
    noise_data = cv2.imread(n_path, cv2.IMREAD_GRAYSCALE)
    # 加载压缩后的图像
    # 计算MSE
    # for x in range(255):
    #     for y in range(255):
    #         if clear_data[x,y]==255:
    #             count+=1


    mse = np.mean((clear_data - noise_data) ** 2)

    # 计算PSNR
    psnr = 10 * np.log10(255 ** 2 / mse)

    print('image{}.png PSNR: {:.2f} dB'.format(idx, psnr))
    psnrs.append(psnr)

# 计算平均PSNR并输出结果
mean_psnr = np.mean(psnrs)
print('平均PSNR: {:.2f} dB'.format(mean_psnr))
