import numpy as np
import cv2


train_noise_4040_path = '/media/jnu/data2/basedata/phase_YH_squared_train4040/'
clear_imgs_train_data = []
noise_imgs_train_data = []
total_value = 0
num=19201
# for idx in range((index-1)*batchsize+1, index*batchsize+1):
for idx in range(1,num):
    # get the match imgs of clears and noises
    n_path = ('%s/%d.png' % (train_noise_4040_path, idx))
    # append clear imgs and noise imgs
    noise_data = cv2.imread(n_path, cv2.IMREAD_GRAYSCALE)
    noise_data =np.array(noise_data)
    noise_imgs_train_data = noise_data / 255
    value = np.mean(noise_imgs_train_data)
    total_value= value+total_value

avarage_value = total_value / (num-1)
print('avarage_value：',avarage_value)



