import numpy as np
import cv2
import glob as glob_module  # 将模块名修改为 glob_module
import torch

# train_clear_6464_path = '/media/jnu/data2/basedata/ampSLM_XH_train_6464'
# train_noise_6464_path = '/media/jnu/data2/basedata/ampSLM_YH_squared_train6464'
train_clear_6464_path = '/media/jnu/1eb5aebb-d8ad-4c4a-a83d-4a86dd7d859a/one_paper_data/dataset/one/HQ'
train_noise_6464_path = '/media/jnu/1eb5aebb-d8ad-4c4a-a83d-4a86dd7d859a/one_paper_data/dataset/one/LQ'

test_clear_6464_path = '/media/jnu/data2/basedata/ampSLM_XH_test_6464'
test_noise_6464_path = '/media/jnu/data2/basedata/ampSLM_YH_squared_test6464'


train_clear_mutimode_6464_path = '/media/jnu/data2/Data_325/dataset/train/original'
train_noise_mutimode_6464_path = '/media/jnu/data2/Data_325/dataset/train/speckle'

test_clear_mutimode_6464_path = '/media/jnu/data2/Data_325/dataset/test_data/original'
test_noise_mutimode_6464_path = '/media/jnu/data2/Data_325/dataset/test_data/speckle'


# train_clear_6464_path = '/media/jnu/1eb5aebb-d8ad-4c4a-a83d-4a86dd7d859a/one_paper_data/dataset/three/HQ'
# train_noise_6464_path = '/media/jnu/1eb5aebb-d8ad-4c4a-a83d-4a86dd7d859a/one_paper_data/dataset/three/LQ'

# global trian_num
# global val_endnum
#
# trian_num = 35000
# val_endnum = 35076



# def load_train_data(batchsize , index):
def load_train_data(lst_epoch):
    clear_imgs_train_data = []
    noise_imgs_train_data = []
    # for idx in range((index-1)*batchsize+1, index*batchsize+1):
    for idx in lst_epoch:
        c_path = ('%s/%d.png' % (train_clear_6464_path, idx))
        n_path = ('%s/%d.png' % (train_noise_6464_path, idx))
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

def load_muti_mode_train_data(lst_epoch):
    clear_imgs_train_data = []
    noise_imgs_train_data = []
    # for idx in range((index-1)*batchsize+1, index*batchsize+1):
    for idx in lst_epoch:
        c_path = ('%s/%d.png' % (train_clear_mutimode_6464_path, idx))
        n_path = ('%s/%d.png' % (train_noise_mutimode_6464_path, idx))
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


# train_clear_6464_path_128_128 = "/media/jnu/data2/xwx/dataset/128_128/HQ"
# train_noise_6464_path_128_128 = "/media/jnu/data2/xwx/dataset/128_128/LQ"
train_clear_6464_path_128_128 = "/media/jnu/data2/simulation_data/128_128/train/HQ"
train_noise_6464_path_128_128 = "/media/jnu/data2/simulation_data/128_128/train/LQ"


def load_train_data_128(lst_epoch):
    clear_imgs_train_data = []
    noise_imgs_train_data = []
    # for idx in range((index-1)*batchsize+1, index*batchsize+1):
    for idx in lst_epoch:
        c_path = ('%s/%d.png' % (train_clear_6464_path_128_128, idx))
        n_path = ('%s/%d.png' % (train_noise_6464_path_128_128, idx))
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


# train_clear_6464_path_32_32 = "/media/jnu/data2/xwx/dataset/32_32/HQ"
# train_noise_6464_path_32_32 = "/media/jnu/data2/xwx/dataset/32_32/SP"

train_clear_6464_path_32_32 = "/media/jnu/data2/simulation_data/32_32/train/HQ"
train_noise_6464_path_32_32 = "/media/jnu/data2/simulation_data/32_32/train/LQ"

def load_train_data_128_32(lst_epoch):
    clear_imgs_train_data = []
    noise_imgs_train_data = []
    # for idx in range((index-1)*batchsize+1, index*batchsize+1):
    for idx in lst_epoch:
        c_path = ('%s/%d.png' % (train_clear_6464_path_32_32, idx))
        n_path = ('%s/%d.png' % (train_noise_6464_path_32_32, idx))
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

# train_clear_6464_path_64_64 = "/media/jnu/data2/xwx/dataset/64_64/HQ"
# train_noise_6464_path_64_64 = "/media/jnu/data2/xwx/dataset/64_64/SP"

train_clear_6464_path_64_64 = "/media/jnu/data2/simulation_data/64_64/train/HQ"
train_noise_6464_path_64_64 = "/media/jnu/data2/simulation_data/64_64/train/LQ"


def load_train_data_128_64(lst_epoch):
    clear_imgs_train_data = []
    noise_imgs_train_data = []
    # for idx in range((index-1)*batchsize+1, index*batchsize+1):
    for idx in lst_epoch:
        c_path = ('%s/%d.png' % (train_clear_6464_path_64_64, idx))
        n_path = ('%s/%d.png' % (train_noise_6464_path_64_64, idx))
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

train_clear_6464_path_100_100 = "/media/jnu/data2/xwx/simulation_data/100_100/train/HQ"
train_noise_6464_path_100_100 = "/media/jnu/data2/xwx/simulation_data/100_100/train/LQ"


def load_train_data_128_100(lst_epoch):
    clear_imgs_train_data = []
    noise_imgs_train_data = []
    # for idx in range((index-1)*batchsize+1, index*batchsize+1):
    for idx in lst_epoch:
        c_path = ('%s/%d.png' % (train_clear_6464_path_100_100, idx))
        n_path = ('%s/%d.png' % (train_noise_6464_path_100_100, idx))
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



train_clear_6464_path_80_80 = "/media/jnu/data2/xwx/simulation_data/80_80/train/HQ"
train_noise_6464_path_80_80 =  "/media/jnu/data2/xwx/simulation_data/80_80/train/LQ"


def load_train_data_128_80(lst_epoch):
    clear_imgs_train_data = []
    noise_imgs_train_data = []
    # for idx in range((index-1)*batchsize+1, index*batchsize+1):
    for idx in lst_epoch:
        c_path = ('%s/%d.png' % (train_clear_6464_path_80_80, idx))
        n_path = ('%s/%d.png' % (train_noise_6464_path_80_80, idx))
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



trian_num =48900
val_endnum =49000
def load_val_data():
    val_clear = []
    val_noise = []
    # for idx in range((index-1)*batchsize+1, index*batchsize+1):
    for idx in range(trian_num+1, val_endnum+1):
        # get the match imgs of clears and noises
        c_path = ('%s/%d.png' % (train_clear_6464_path, idx))
        n_path = ('%s/%d.png' % (train_noise_6464_path, idx))
        # append clear imgs and noise imgs
        clear_data = cv2.imread(c_path, cv2.IMREAD_GRAYSCALE)
        noise_data = cv2.imread(n_path, cv2.IMREAD_GRAYSCALE)
        # append clear imgs and noise imgs
        val_clear.append(clear_data)
        val_noise.append(noise_data)
    val_clear = np.array(val_clear)
    val_noise = np.array(val_noise)
    val_clear = val_clear / 255.0
    val_noise = val_noise / 255.0
    val_clear = torch.tensor(val_clear).unsqueeze(1)
    val_noise = torch.tensor(val_noise).unsqueeze(1)

    return val_clear , val_noise


# test_clear_6464_path = '/media/jnu/data1/4_20_shiyan/jujiaotest/yuce/testepoch1'
# test_clear_6464_path = '/media/jnu/data1/5_8_shiyan/20000_quyujujiao/LQ/testepoch1'
def load_test_data():

    test_noise = []
    test_clear = []

    for idx in range(1, 7):
        # get the match imgs of clears and noises
        c_path = ('%s/%d.png' % (test_clear_6464_path, idx))
        n_path = ('%s/%d.png' % (test_noise_6464_path, idx))
        # append clear imgs and noise imgs
        clear_data = cv2.imread(c_path, cv2.IMREAD_GRAYSCALE)
        noise_data = cv2.imread(n_path, cv2.IMREAD_GRAYSCALE)
        # append clear imgs and noise imgs
        test_clear.append(clear_data)
        test_noise.append(noise_data)
    test_clear = np.array(test_clear)
    test_noise = np.array(test_noise)
    test_clear = test_clear / 255.0
    test_noise = test_noise / 255.0

    test_clear = torch.tensor(test_clear).unsqueeze(1)
    test_noise = torch.tensor(test_noise).unsqueeze(1)
    return test_clear , test_noise

def load_mutimode_test_data():

    test_noise = []
    test_clear = []

    for idx in range(1, 5):
        # get the match imgs of clears and noises
        c_path = ('%s/%d.png' % (test_clear_mutimode_6464_path, idx))
        n_path = ('%s/%d.png' % (test_noise_mutimode_6464_path, idx))
        # append clear imgs and noise imgs
        clear_data = cv2.imread(c_path, cv2.IMREAD_GRAYSCALE)
        noise_data = cv2.imread(n_path, cv2.IMREAD_GRAYSCALE)
        # append clear imgs and noise imgs
        test_clear.append(clear_data)
        test_noise.append(noise_data)
    test_clear = np.array(test_clear)
    test_noise = np.array(test_noise)
    test_clear = test_clear / 255.0
    test_noise = test_noise / 255.0

    test_clear = torch.tensor(test_clear).unsqueeze(1)
    test_noise = torch.tensor(test_noise).unsqueeze(1)
    return test_clear , test_noise


test_clear_6464_path128_128 = "/media/jnu/data2/xwx/dataset/128_128/HQ"
test_noise_6464_path128_128 = "/media/jnu/data2/xwx/dataset/128_128/SP"



def load_test_data128_128():

    test_noise = []
    test_clear = []

    for idx in range(9981, 10001):
        # get the match imgs of clears and noises
        c_path = ('%s/%d.png' % (test_clear_6464_path128_128, idx))
        n_path = ('%s/%d.png' % (test_noise_6464_path128_128, idx))
        # append clear imgs and noise imgs
        clear_data = cv2.imread(c_path, cv2.IMREAD_GRAYSCALE)
        noise_data = cv2.imread(n_path, cv2.IMREAD_GRAYSCALE)
        # append clear imgs and noise imgs
        test_clear.append(clear_data)
        test_noise.append(noise_data)
    test_clear = np.array(test_clear)
    test_noise = np.array(test_noise)
    test_clear = test_clear / 255.0
    test_noise = test_noise / 255.0

    test_clear = torch.tensor(test_clear).unsqueeze(1)
    test_noise = torch.tensor(test_noise).unsqueeze(1)
    return test_clear , test_noise


# test_clear_6464_path32_32 = "/media/jnu/data2/xwx/dataset/32_32/HQ"
# test_noise_6464_path32_32 = "/media/jnu/data2/xwx/dataset/32_32/SP"

test_clear_6464_path32_32 = "/media/jnu/data2/xwx/GS/32/face/our_face_HQ"
test_noise_6464_path32_32 = "/media/jnu/data2/xwx/GS/32/face/our_face_LQ"
def load_test_data128_32():

    test_noise = []
    test_clear = []

    for idx in range(5, 8):
        # get the match imgs of clears and noises
        c_path = ('%s/%d.png' % (test_clear_6464_path32_32, idx))
        n_path = ('%s/%d.png' % (test_noise_6464_path32_32, idx))
        # append clear imgs and noise imgs
        clear_data = cv2.imread(c_path, cv2.IMREAD_GRAYSCALE)
        noise_data = cv2.imread(n_path, cv2.IMREAD_GRAYSCALE)
        # append clear imgs and noise imgs
        test_clear.append(clear_data)
        test_noise.append(noise_data)
    test_clear = np.array(test_clear)
    test_noise = np.array(test_noise)
    test_clear = test_clear / 255.0
    test_noise = test_noise / 255.0

    test_clear = torch.tensor(test_clear).unsqueeze(1)
    test_noise = torch.tensor(test_noise).unsqueeze(1)
    return test_clear , test_noise

# test_clear_6464_path64_64 = "/media/jnu/data2/xwx/dataset/64_64/HQ"
# test_noise_6464_path64_64  = "/media/jnu/data2/xwx/dataset/64_64/SP"
test_clear_6464_path64_64 = "/media/jnu/data2/xwx/GS/64/face/our_face_HQ"
test_noise_6464_path64_64 = "/media/jnu/data2/xwx/GS/64/face/our_face_LQ"

# test_clear_6464_path64_64 = "/media/jnu/data2/simulation_data/64_64/val/HQ"
# test_noise_6464_path64_64  = "/media/jnu/data2/simulation_data/64_64/val/LQ"


def load_test_data128_64():

    test_noise = []
    test_clear = []

    for idx in range(5, 8):
        # get the match imgs of clears and noises
        c_path = ('%s/%d.png' % (test_clear_6464_path64_64 , idx))
        n_path = ('%s/%d.png' % (test_noise_6464_path64_64 , idx))
        # append clear imgs and noise imgs
        clear_data = cv2.imread(c_path, cv2.IMREAD_GRAYSCALE)
        noise_data = cv2.imread(n_path, cv2.IMREAD_GRAYSCALE)
        # append clear imgs and noise imgs
        test_clear.append(clear_data)
        test_noise.append(noise_data)
    test_clear = np.array(test_clear)
    test_noise = np.array(test_noise)
    test_clear = test_clear / 255.0
    test_noise = test_noise / 255.0

    test_clear = torch.tensor(test_clear).unsqueeze(1)
    test_noise = torch.tensor(test_noise).unsqueeze(1)
    return test_clear , test_noise



# test_clear_6464_path64_64 = "/media/jnu/data2/xwx/dataset/64_64/HQ"
# test_noise_6464_path64_64  = "/media/jnu/data2/xwx/dataset/64_64/SP"
test_clear_6464_path100_100 = "/media/jnu/data2/xwx/GS/64/face/our_face_HQ"
test_noise_6464_path100_100 = "/media/jnu/data2/xwx/GS/64/face/our_face_LQ"

def load_test_data128_100():

    test_noise = []
    test_clear = []

    for idx in range(1, 8):
        # get the match imgs of clears and noises
        c_path = ('%s/%d.png' % (test_clear_6464_path100_100 , idx))
        n_path = ('%s/%d.png' % (test_noise_6464_path100_100 , idx))
        # append clear imgs and noise imgs
        clear_data = cv2.imread(c_path, cv2.IMREAD_GRAYSCALE)
        noise_data = cv2.imread(n_path, cv2.IMREAD_GRAYSCALE)
        # append clear imgs and noise imgs
        test_clear.append(clear_data)
        test_noise.append(noise_data)
    test_clear = np.array(test_clear)
    test_noise = np.array(test_noise)
    test_clear = test_clear / 255.0
    test_noise = test_noise / 255.0

    test_clear = torch.tensor(test_clear).unsqueeze(1)
    test_noise = torch.tensor(test_noise).unsqueeze(1)
    return test_clear , test_noise


test_clear_80_80_path80_80 = "/media/jnu/data2/xwx/GS/80/face/our_face_HQ"
test_noise_80_80_path80_80 = "/media/jnu/data2/xwx/GS/80/face/our_face_LQ"

# test_clear_6464_path64_64 = "/media/jnu/data2/simulation_data/64_64/val/HQ"
# test_noise_6464_path64_64  = "/media/jnu/data2/simulation_data/64_64/val/LQ"


def load_test_data128_80():

    test_noise = []
    test_clear = []

    for idx in range(1, 10):
        # get the match imgs of clears and noises
        c_path = ('%s/%d.png' % (test_clear_80_80_path80_80 , idx))
        n_path = ('%s/%d.png' % (test_noise_80_80_path80_80 , idx))
        # append clear imgs and noise imgs
        clear_data = cv2.imread(c_path, cv2.IMREAD_GRAYSCALE)
        noise_data = cv2.imread(n_path, cv2.IMREAD_GRAYSCALE)
        # append clear imgs and noise imgs
        test_clear.append(clear_data)
        test_noise.append(noise_data)
    test_clear = np.array(test_clear)
    test_noise = np.array(test_noise)
    test_clear = test_clear / 255.0
    test_noise = test_noise / 255.0

    test_clear = torch.tensor(test_clear).unsqueeze(1)
    test_noise = torch.tensor(test_noise).unsqueeze(1)
    return test_clear , test_noise





# test_noise_6464_path64_64  = "/media/jnu/data2/xwx/dataset/64_64/SP"
test_clear_6464_path100_100 = "/media/jnu/data2/xwx/GS/100/face/our_face_HQ"
test_noise_6464_path100_100 = "/media/jnu/data2/xwx/GS/100/face/our_face_LQ"


def load_test_data128_100():

    test_noise = []
    test_clear = []

    for idx in range(1, 10):
        # get the match imgs of clears and noises
        c_path = ('%s/%d.png' % (test_clear_6464_path100_100 , idx))
        n_path = ('%s/%d.png' % (test_noise_6464_path100_100 , idx))
        # append clear imgs and noise imgs
        clear_data = cv2.imread(c_path, cv2.IMREAD_GRAYSCALE)
        noise_data = cv2.imread(n_path, cv2.IMREAD_GRAYSCALE)
        # append clear imgs and noise imgs
        test_clear.append(clear_data)
        test_noise.append(noise_data)
    test_clear = np.array(test_clear)
    test_noise = np.array(test_noise)
    test_clear = test_clear / 255.0
    test_noise = test_noise / 255.0

    test_clear = torch.tensor(test_clear).unsqueeze(1)
    test_noise = torch.tensor(test_noise).unsqueeze(1)
    return test_clear , test_noise

# test_clear_6464_path128_128face = "/media/jnu/data2/xwx/dataset/128_128/new_data/HQ"
# test_noise_6464_path128_128face = "/media/jnu/data2/xwx/dataset/128_128/new_data/SP"

test_clear_6464_path128_128face = "/media/jnu/data2/xwx/GS/128/face/our_face_HQ"
test_noise_6464_path128_128face = "/media/jnu/data2/xwx/GS/128/face/our_face_LQ"

def load_test_data128_128face():

    test_noise = []
    test_clear = []

    for idx in range(5, 8):
        # get the match imgs of clears and noises
        c_path = ('%s/%d.png' % (test_clear_6464_path128_128face, idx))
        n_path = ('%s/%d.png' % (test_noise_6464_path128_128face, idx))
        # append clear imgs and noise imgs
        clear_data = cv2.imread(c_path, cv2.IMREAD_GRAYSCALE)
        noise_data = cv2.imread(n_path, cv2.IMREAD_GRAYSCALE)
        # append clear imgs and noise imgs
        test_clear.append(clear_data)
        test_noise.append(noise_data)
    test_clear = np.array(test_clear)
    test_noise = np.array(test_noise)
    test_clear = test_clear / 255.0
    test_noise = test_noise / 255.0

    test_clear = torch.tensor(test_clear).unsqueeze(1)
    test_noise = torch.tensor(test_noise).unsqueeze(1)
    return test_clear , test_noise

# test_noise_6464_path_1 = '/media/jnu/data1/4_20_shiyan/mid_test_2000_dual/testepoch1'
# test_clear_6464_path_1 = '/media/jnu/data1/4_20_shiyan/jujiaotrue/'
test_clear_6464_path_1 = '/media/jnu/data2/basedata/ampSLM_XH_test_6464'
# test_noise_6464_path_1 = '/media/jnu/data1/4_20_shiyan/mat_to_mat_test/mat_sanban'
#
# test_clear_6464_path = '/media/jnu/data2/basedata/ampSLM_XH_test_6464'
test_noise_6464_path_1 = '/media/jnu/data2/basedata/ampSLM_YH_squared_test6464'

# test_noise_6464_path_1 = '/media/jnu/data1/4_20_shiyan/2000_net/mat_to_20000_net_test/mat_sanban'
# test_noise_6464_path_1 = '/media/jnu/data1/5_8_shiyan/20000_quyujujiao/HQ'

def load_test_data_1():
    test_noise = []
    test_clear = []

    for idx in range(1, 7):
        # get the match imgs of clears and noises
        c_path = ('%s/%d.png' % (test_clear_6464_path_1, idx))
        n_path = ('%s/%d.png' % (test_noise_6464_path_1, idx))
        # append clear imgs and noise imgs
        clear_data = cv2.imread(c_path, cv2.IMREAD_GRAYSCALE)
        noise_data = cv2.imread(n_path, cv2.IMREAD_GRAYSCALE)
        # append clear imgs and noise imgs
        test_clear.append(clear_data)
        test_noise.append(noise_data)
    test_clear = np.array(test_clear)
    test_noise = np.array(test_noise)
    test_clear = test_clear / 255.0
    test_noise = test_noise / 255.0

    test_clear = torch.tensor(test_clear).unsqueeze(1)
    test_noise = torch.tensor(test_noise).unsqueeze(1)
    return test_clear , test_noise


model_flag = 3
test_minist_clear_6464_path = '/media/jnu/data2/basedata/phase_YH_squared_train4040'
# test_minist_noise_6464_path = '/media/jnu/data1/focusdata/net/GT'
test_minist_noise_6464_path = '/media/jnu/data1/TM_test/net/testclear1'


def load_minist_test_data():

    test_noise = []
    test_clear = []

    for idx in range(1, 6):
        # get the match imgs of clears and noises
        if model_flag == 1:
            c_path = ('%s/%d.png' % (test_clear_1616_path, idx))
            n_path = ('%s/%d.png' % (test_noise_1616_path, idx))
        elif model_flag == 2:
            c_path = ('%s/%d.png' % (test_clear_4040_path, idx))
            n_path = ('%s/%d.png' % (test_noise_4040_path, idx))
        elif model_flag == 3:
            c_path = ('%s/%d.png' % (test_minist_clear_6464_path, idx))
            n_path = ('%s/%d.png' % (test_minist_noise_6464_path, idx))
        elif model_flag == 4:
            c_path = ('%s/%d.png' % (test_clear_2828_path, idx))
            n_path = ('%s/%d.png' % (test_noise_2828_path, idx))
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


# test_minist_clear_6464_path_plus = '/media/jnu/data2/cyclenet_test_result/Linear_conv6464/minist/testnoise1/'
# test_minist_noise_6464_path_plus = '/media/jnu/data2/cyclenet_test_result/Linear_conv6464/minist/testclear1/'
# test_minist_clear_6464_path_plus = '/media/jnu/data1/TM_test/net_5000_dot/GT/'
# test_minist_noise_6464_path_plus = '/media/jnu/data1/TM_test/net_5000_dot/LQ/'
#

# test_minist_clear_6464_path_plus = '/media/jnu/data1/now_shiyan/true/'
# test_minist_noise_6464_path_plus = '/media/jnu/data1/TM_test/net_5000_dot/LQ/'

# test_minist_clear_6464_path_plus = '/media/jnu/data1/now_shiyan/jiaochayanzheng/true/'
# test_minist_noise_6464_path_plus = '/media/jnu/data1/TM_test/net_5000_dot/LQ/'
# test_minist_clear_6464_path_plus = '/media/jnu/data1/now_shiyan2/20000_2000/testmodel_clear1/'
# test_minist_noise_6464_path_plus = '/media/jnu/data1/TM_test/net_500_dot/LQ'
# test_minist_clear_6464_path_plus = '/media/jnu/data1/jujiao_net/jujiaonet20000/testmodel_clear1'
# test_minist_noise_6464_path_plus = '/media/jnu/data1/TM_test/net_5000_dot/LQ/'

# test_minist_clear_6464_path_plus = '/media/jnu/data2/basedata/ampSLM_XH_test_6464'
# test_minist_noise_6464_path_plus = '/media/jnu/data2/basedata/ampSLM_YH_squared_test6464'


# test_minist_clear_6464_path_plus = '/media/jnu/data1/4_20_shiyan/2000_net/net_to_mat_mnist/HQ'
# test_minist_noise_6464_path_plus = '/media/jnu/data1/TM_test/net_5000_dot/LQ/'

# test_minist_clear_6464_path_plus = '/media/jnu/data1/4_20_shiyan/2000_net/2000_net_to_20000_net_mnist/HQ'
# test_minist_noise_6464_path_plus = '/media/jnu/data2/basedata/ampSLM_YH_squared_test6464'
# # test_minist_noise_6464_path_plus = '/home/jnu/文档/sigmoid/ampSLM_YH_squared_test6464'
# # test_minist_noise_6464_path_plus = '/media/jnu/data2/10_29/focus/dot'
# # test_minist_clear_6464_path_plus = '/media/jnu/1eb5aebb-d8ad-4c4a-a83d-4a86dd7d859a/one_paper_data/paper_data/TM/mnist/JT_net/HQ'
# test_minist_clear_6464_path_plus = '/media/jnu/data2/basedata/ampSLM_XH_test_6464'

# test_minist_clear_6464_path_plus = '/media/jnu/data1/4_20_shiyan/2000_net/2000_net_to_20000_net_mnist/HQ'
test_minist_noise_6464_path_plus = '/home/jnu/视频/SP'
# test_minist_noise_6464_path_plus = '/home/jnu/文档/sigmoid/ampSLM_YH_squared_test6464'
# test_minist_noise_6464_path_plus = '/media/jnu/data2/10_29/focus/dot'
# test_minist_clear_6464_path_plus = '/media/jnu/1eb5aebb-d8ad-4c4a-a83d-4a86dd7d859a/one_paper_data/paper_data/TM/mnist/JT_net/HQ'
test_minist_clear_6464_path_plus = '/home/jnu/视频/HQ'


def load_minist_test_data_plus():

    test_noise = []
    test_clear = []

    for idx in range(1, 16):
        # get the match imgs of clears and noises
        c_path = ('%s/%d.png' % (test_minist_clear_6464_path_plus, idx))
        n_path = ('%s/%d.png' % (test_minist_noise_6464_path_plus, idx))
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

test_minist_noise_6464_path_1024 = '/media/jnu/data2/model_3_14/true'
# test_minist_clear_6464_path_plus = '/media/jnu/1eb5aebb-d8ad-4c4a-a83d-4a86dd7d859a/one_paper_data/paper_data/TM/mnist/JT_net/HQ'
test_minist_clear_6464_path_1024 = '/media/jnu/data2/model_12_9/focus/HQ128'

# test_minist_clear_6464_path_1024 = '/media/jnu/data2/basedata/ampSLM_XH_train_6464'

def load_minist_test_data_1024():

    test_noise = []
    test_clear = []

    for idx in range(6, 7):
        # get the match imgs of clears and noises
        c_path = ('%s/%d.png' % (test_minist_clear_6464_path_1024, idx))
        n_path = ('%s/%d.png' % (test_minist_noise_6464_path_1024, idx))
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



test_minist_noise_6464_path_focus_128_128 = '/media/jnu/data2/model_12_9/focus/LQ_1'
# test_minist_clear_6464_path_plus = '/media/jnu/1eb5aebb-d8ad-4c4a-a83d-4a86dd7d859a/one_paper_data/paper_data/TM/mnist/JT_net/HQ'
test_minist_clear_6464_path_focus_128_128  = '/media/jnu/data2/model_12_9/focus/HQ128'

# test_minist_clear_6464_path_1024 = '/media/jnu/data2/basedata/ampSLM_XH_train_6464'

def load_minist_test_data_focus_128_128():

    test_noise = []
    test_clear = []

    for idx in range(1, 7):
        # get the match imgs of clears and noises
        c_path = ('%s/%d.png' % (test_minist_clear_6464_path_focus_128_128, idx))
        n_path = ('%s/%d.png' % (test_minist_noise_6464_path_focus_128_128, idx))
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


test_minist_noise_6464_path_focus_64_64  = '/media/jnu/data2/model_12_9/focus/LQ'
# test_minist_clear_6464_path_plus = '/media/jnu/1eb5aebb-d8ad-4c4a-a83d-4a86dd7d859a/one_paper_data/paper_data/TM/mnist/JT_net/HQ'
test_minist_clear_6464_path_focus_64_64  = '/media/jnu/data2/model_12_9/focus/HQ64'

# test_minist_clear_6464_path_1024 = '/media/jnu/data2/basedata/ampSLM_XH_train_6464'

def load_minist_test_data_focus_64_64():

    test_noise = []
    test_clear = []

    for idx in range(1, 28):
        # get the match imgs of clears and noises
        c_path = ('%s/%d.png' % (test_minist_clear_6464_path_focus_64_64, idx))
        n_path = ('%s/%d.png' % (test_minist_noise_6464_path_focus_64_64, idx))
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

test_minist_noise_6464_path_focus_32_32 = '/media/jnu/data2/model_12_9/focus/LQ_1'
# test_minist_clear_6464_path_plus = '/media/jnu/1eb5aebb-d8ad-4c4a-a83d-4a86dd7d859a/one_paper_data/paper_data/TM/mnist/JT_net/HQ'
test_minist_clear_6464_path_focus_32_32  = '/media/jnu/data2/model_12_9/focus/HQ128'

# test_minist_clear_6464_path_1024 = '/media/jnu/data2/basedata/ampSLM_XH_train_6464'

def load_minist_test_data_focus_32_32():

    test_noise = []
    test_clear = []

    for idx in range(1, 28):
        # get the match imgs of clears and noises
        c_path = ('%s/%d.png' % (test_minist_clear_6464_path_focus_32_32, idx))
        n_path = ('%s/%d.png' % (test_minist_noise_6464_path_focus_32_32, idx))
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


test_minist_clear_6464_path_plus_jujiao_weitiao = '/media/jnu/data2/basedata/ampSLM_YH_squared_train6464'
# test_minist_clear_6464_path_plus = '/media/jnu/SUCCESS/data_5_29/jujiao/prvamp_jujiao/dian/HQ'
# test_minist_noise_6464_path_plus_jujiao_weitiao =  '/media/jnu/1eb5aebb-d8ad-4c4a-a83d-4a86dd7d859a/one_paper_data/dataset/one/HQ256256'
test_minist_noise_6464_path_plus_jujiao_weitiao = '/media/jnu/data2/10_29/focus/dot'

def load_minist_test_jujiao():

    test_noise = []
    test_clear = []

    for idx in range(1, 7):
        # get the match imgs of clears and noises

        c_path = ('%s/%d.png' % (test_minist_clear_6464_path_plus_jujiao_weitiao, idx))
        n_path = ('%s/%d.png' % (test_minist_noise_6464_path_plus_jujiao_weitiao, idx))

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



test_minist_clear_6464_path_test_jujiao = '/media/jnu/data2/basedata/ampSLM_YH_squared_train6464'
# test_minist_clear_6464_path_plus = '/media/jnu/SUCCESS/data_5_29/jujiao/prvamp_jujiao/dian/HQ'
# test_minist_noise_6464_path_plus_jujiao_weitiao =  '/media/jnu/1eb5aebb-d8ad-4c4a-a83d-4a86dd7d859a/one_paper_data/dataset/one/HQ256256'
test_minist_noise_6464_path_test_jujiao = '/media/jnu/data2/model18jujiao_weitiao/LQ_2'

def load_minist_test_data_jujiao_test_jujiao():

    test_noise = []
    test_clear = []

    for idx in range(1, 7):
        # get the match imgs of clears and noises
        if model_flag == 1:
            c_path = ('%s/%d.png' % (test_clear_1616_path, idx))
            n_path = ('%s/%d.png' % (test_noise_1616_path, idx))
        elif model_flag == 2:
            c_path = ('%s/%d.png' % (test_clear_4040_path, idx))
            n_path = ('%s/%d.png' % (test_noise_4040_path, idx))
        elif model_flag == 3:
            c_path = ('%s/%d.png' % (test_minist_clear_6464_path_test_jujiao, idx))
            n_path = ('%s/%d.png' % (test_minist_noise_6464_path_test_jujiao, idx))
        elif model_flag == 4:
            c_path = ('%s/%d.png' % (test_clear_2828_path, idx))
            n_path = ('%s/%d.png' % (test_noise_2828_path, idx))
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


train_minist_clear_6464_path_plus_jujiao_weitiao = '/media/jnu/data2/basedata/ampSLM_YH_squared_train6464'
# test_minist_clear_6464_path_plus = '/media/jnu/SUCCESS/data_5_29/jujiao/prvamp_jujiao/dian/HQ'
train_minist_noise_6464_path_plus_jujiao_weitiao =  '/media/jnu/1eb5aebb-d8ad-4c4a-a83d-4a86dd7d859a/one_paper_data/dataset/one/HQ256256'

def load_minist_train_data_plus_jujiao_weitiao():

    test_noise = []
    test_clear = []

    for idx in range(1, 1001):
        # get the match imgs of clears and noises
        if model_flag == 1:
            c_path = ('%s/%d.png' % (test_clear_1616_path, idx))
            n_path = ('%s/%d.png' % (test_noise_1616_path, idx))
        elif model_flag == 2:
            c_path = ('%s/%d.png' % (test_clear_4040_path, idx))
            n_path = ('%s/%d.png' % (test_noise_4040_path, idx))
        elif model_flag == 3:
            c_path = ('%s/%d.png' % (train_minist_clear_6464_path_plus_jujiao_weitiao, idx))
            n_path = ('%s/%d.png' % (train_minist_noise_6464_path_plus_jujiao_weitiao, idx))
        elif model_flag == 4:
            c_path = ('%s/%d.png' % (test_clear_2828_path, idx))
            n_path = ('%s/%d.png' % (test_noise_2828_path, idx))
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
# test_matfocus_clear_6464_path_plus = '/media/jnu/data1/focusdata/net/GT'  #反向
# test_matfocus_noise_6464_path_plus = '/media/jnu/data1/focusdata/net/LQ'

# test_matfocus_clear_6464_path_plus = '/media/jnu/data1/focus_test/net_4/GT'  #正向
# test_matfocus_noise_6464_path_plus = '/media/jnu/data1/focus_test/net_4/LQ'

# test_matfocus_clear_6464_path_plus = '/media/jnu/data1/TM_test/net_500_dot/GT'  #正向
# test_matfocus_noise_6464_path_plus = '/media/jnu/data1/TM_test/net_500_dot/LQ'

# test_matfocus_clear_6464_path_plus = '/media/jnu/data1/TM_test/net_500_dot/GT'  #正向
# test_matfocus_noise_6464_path_plus = '/media/jnu/data1/now_shiyan/5000_dubbin/testmodel_clear1/'
# test_matfocus_noise_6464_path_plus = '/media/jnu/data1/TM_test/net_500_dot/GT'
# test_matfocus_clear_6464_path_plus = '/media/jnu/data1/TM_test/net_500_dot/GT'  #正向
# test_matfocus_noise_6464_path_plus = '/media/jnu/data1/4_20_shiyan/mid_test_2000/yuce/testmodel_clear1'
# test_matfocus_noise_6464_path_plus = '/media/jnu/data1/TM_test/net_500_dot/GT'
# test_matfocus_clear_6464_path_plus = '/media/jnu/data1/4_20_shiyan/2000_net/net_to_mat_mnist/HQ'
# test_matfocus_clear_6464_path_plus = '/media/jnu/data2/basedata/ampSLM_XH_test_6464'
# test_matfocus_noise_6464_path_plus = '/media/jnu/data2/basedata/ampSLM_YH_squared_test6464'
# test_matfocus_clear_6464_path_plus = '/media/jnu/data1/4_20_shiyan/net_B_20000_mat_20000_mnist/HQ'
# test_matfocus_noise_6464_path_plus = '/media/jnu/data1/4_20_shiyan/2000_net/mat_to_20000_net_test/mat_sanban'

test_matfocus_clear_6464_path_plus = '/media/jnu/data2/basedata/ampSLM_XH_test_6464'
# test_matfocus_noise_6464_path_plus = '/media/jnu/data2/basedata/ampSLM_YH_squared_test6464'

# test_matfocus_noise_6464_path_plus = '/media/jnu/1eb5aebb-d8ad-4c4a-a83d-4a86dd7d859a/one_paper_data' \
#                                      '/paper_data/TM/TM_cross/prvamp_net/LQ'
# test_matfocus_noise_6464_path_plus =  '/media/jnu/1eb5aebb-d8ad-4c4a-a83d-4a86dd7d859a/one_paper_data/paper_data/TM/TM_cross/prvamp_net/LQ'

test_matfocus_noise_6464_path_plus = '/media/jnu/data2/prvamp_add_noise/LQ_4/'
def load_matfocus_test_data_plus():

    test_noise = []
    test_clear = []

    for idx in range(1,7):
        # get the match imgs of clears and noises
        if model_flag == 1:
            c_path = ('%s/%d.png' % (test_clear_1616_path, idx))
            n_path = ('%s/%d.png' % (test_noise_1616_path, idx))
        elif model_flag == 2:
            c_path = ('%s/%d.png' % (test_clear_4040_path, idx))
            n_path = ('%s/%d.png' % (test_noise_4040_path, idx))
        elif model_flag == 3:
            c_path = ('%s/%d.png' % (test_matfocus_clear_6464_path_plus, idx))
            n_path = ('%s/%d.png' % (test_matfocus_noise_6464_path_plus, idx))
        elif model_flag == 4:
            c_path = ('%s/%d.png' % (test_clear_2828_path, idx))
            n_path = ('%s/%d.png' % (test_noise_2828_path, idx))
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

test_clear_focus_6464_path_plus = '/media/jnu/data2/foucs_image/GT/'
test_noise_focus_6464_path_plus = '/media/jnu/data2/foucs_image/LQ/'
def load_focus_test_data():

    test_noise = []
    test_clear = []

    for idx in range(1, 6):
        # get the match imgs of clears and noises
        if model_flag == 1:
            c_path = ('%s/%d.png' % (test_clear_1616_path, idx))
            n_path = ('%s/%d.png' % (test_noise_1616_path, idx))
        elif model_flag == 2:
            c_path = ('%s/%d.png' % (test_clear_4040_path, idx))
            n_path = ('%s/%d.png' % (test_noise_4040_path, idx))
        elif model_flag == 3:
            c_path = ('%s/%d.png' % (test_clear_focus_6464_path_plus, idx))
            n_path = ('%s/%d.png' % (test_noise_focus_6464_path_plus, idx))
        elif model_flag == 4:
            c_path = ('%s/%d.png' % (test_clear_2828_path, idx))
            n_path = ('%s/%d.png' % (test_noise_2828_path, idx))
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


test_clear_focus_6464_path_plus = '/media/jnu/data2/foucs_image/GT/'
test_noise_focus_6464_path_plus = '/media/jnu/data2/model_6_14_new/data/dianjujiaotrue'
def load_focus_train_jujiao_data():

    test_noise = []
    test_clear = []

    for idx in range(1, 6):
        # get the match imgs of clears and noises
        if model_flag == 1:
            c_path = ('%s/%d.png' % (test_clear_1616_path, idx))
            n_path = ('%s/%d.png' % (test_noise_1616_path, idx))
        elif model_flag == 2:
            c_path = ('%s/%d.png' % (test_clear_4040_path, idx))
            n_path = ('%s/%d.png' % (test_noise_4040_path, idx))
        elif model_flag == 3:
            c_path = ('%s/%d.png' % (test_clear_focus_6464_path_plus, idx))
            n_path = ('%s/%d.png' % (test_noise_focus_6464_path_plus, idx))
        elif model_flag == 4:
            c_path = ('%s/%d.png' % (test_clear_2828_path, idx))
            n_path = ('%s/%d.png' % (test_noise_2828_path, idx))
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