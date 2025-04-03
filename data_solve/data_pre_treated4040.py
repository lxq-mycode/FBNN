import numpy as np
import cv2
import glob as glob_module  # 将模块名修改为 glob_module
import torch


train_clear_4040_path = '/media/jnu/data2/basedata/phase_XH_train_4040/'
train_noise_4040_path = '/media/jnu/data2/basedata/phase_YH_squared_train4040/'

test_clear_4040_path = '/media/jnu/data2/basedata/ampSLM_XH_test_4040'
test_noise_4040_path = '/media/jnu/data2/basedata/ampSLM_YH_squared_test4040'





# def load_train_data(batchsize , index):
def load_train_data(lst_epoch):
    clear_imgs_train_data = []
    noise_imgs_train_data = []
    # for idx in range((index-1)*batchsize+1, index*batchsize+1):
    for idx in lst_epoch:
        c_path = ('%s/%d.png' % (train_clear_4040_path, idx))
        n_path = ('%s/%d.png' % (train_noise_4040_path, idx))
        # append clear imgs and noise imgs
        clear_data = cv2.imread(c_path, cv2.IMREAD_GRAYSCALE)
        noise_data = cv2.imread(n_path, cv2.IMREAD_GRAYSCALE)
        # append clear imgs and noise imgs
        clear_imgs_train_data.append(clear_data)
        noise_imgs_train_data.append(noise_data)

    clear_imgs_train_data = np.array(clear_imgs_train_data)
    noise_imgs_train_data = np.array(noise_imgs_train_data)
    clear_imgs_train_data = clear_imgs_train_data / 255
    noise_imgs_train_data = noise_imgs_train_data / 255

    # 将数据转换为 PyTorch 张量
    clear_imgs_train_data= torch.tensor(clear_imgs_train_data).unsqueeze(1)
    noise_imgs_train_data = torch.tensor(noise_imgs_train_data).unsqueeze(1)
    return clear_imgs_train_data , noise_imgs_train_data

# def load_val_data(batchsize , index):



def load_test_data():

    test_noise = []
    test_clear = []

    for idx in range(1, 6):
        # get the match imgs of clears and noises

        c_path = ('%s/%d.png' % (test_clear_4040_path, idx))
        n_path = ('%s/%d.png' % (test_noise_4040_path, idx))

        # append clear imgs and noise imgs
        clear_data = cv2.imread(c_path, cv2.IMREAD_GRAYSCALE)
        noise_data = cv2.imread(n_path, cv2.IMREAD_GRAYSCALE)
        # append clear imgs and noise imgs
        test_clear.append(clear_data)
        test_noise.append(noise_data)
    test_clear = np.array(test_clear)
    test_noise = np.array(test_noise)
    test_clear = test_clear /255
    test_noise = test_noise /255

    test_clear = torch.tensor(test_clear).unsqueeze(1)
    test_noise = torch.tensor(test_noise).unsqueeze(1)
    return test_clear , test_noise


def load_train_data_plus(lst_epoch):
    clear_imgs_train_data = []
    noise_imgs_train_data = []
    # for idx in range((index-1)*batchsize+1, index*batchsize+1):
    for idx in lst_epoch:
        c_path = ('%s/%d.png' % (train_clear_4040_path, idx))
        n_path = ('%s/%d.png' % (train_noise_4040_path, idx))
        # append clear imgs and noise imgs
        clear_data = cv2.imread(c_path, cv2.IMREAD_GRAYSCALE)
        noise_data = cv2.imread(n_path, cv2.IMREAD_GRAYSCALE)
        # append clear imgs and noise imgs
        clear_imgs_train_data.append(clear_data)
        noise_imgs_train_data.append(noise_data)

    clear_imgs_train_data = np.array(clear_imgs_train_data)
    noise_imgs_train_data = np.array(noise_imgs_train_data)
    clear_imgs_train_data = clear_imgs_train_data / 255
    noise_imgs_train_data = noise_imgs_train_data / 255
    clear_imgs_train_data = (clear_imgs_train_data * 2) - 1

    # 将数据转换为 PyTorch 张量
    clear_imgs_train_data= torch.tensor(clear_imgs_train_data).unsqueeze(1)
    noise_imgs_train_data = torch.tensor(noise_imgs_train_data).unsqueeze(1)
    return clear_imgs_train_data , noise_imgs_train_data

# def load_val_data(batchsize , index):



def load_test_data_plus():

    test_noise = []
    test_clear = []

    for idx in range(1, 6):
        # get the match imgs of clears and noises

        c_path = ('%s/%d.png' % (test_clear_4040_path, idx))
        n_path = ('%s/%d.png' % (test_noise_4040_path, idx))

        # append clear imgs and noise imgs
        clear_data = cv2.imread(c_path, cv2.IMREAD_GRAYSCALE)
        noise_data = cv2.imread(n_path, cv2.IMREAD_GRAYSCALE)
        # append clear imgs and noise imgs
        test_clear.append(clear_data)
        test_noise.append(noise_data)
    test_clear = np.array(test_clear)
    test_noise = np.array(test_noise)
    test_clear = test_clear /255
    test_noise = test_noise /255
    test_clear = (test_clear * 2) - 1
    test_clear = torch.tensor(test_clear).unsqueeze(1)
    test_noise = torch.tensor(test_noise).unsqueeze(1)
    return test_clear , test_noise
