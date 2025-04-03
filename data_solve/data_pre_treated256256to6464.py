import numpy as np
import cv2
import glob as glob_module  # 将模块名修改为 glob_module
import torch
train_clear_6464_path = '/media/jnu/data1/data_minist_rice/predata'
train_noise_6464_path = '/media/jnu/data2/basedata/ampSLM_YH_squared_train6464'

val_clear_6464_path = '/media/jnu/data2/basedata/ampSLM_XH_train_6464'
val_noise_6464_path = '/media/jnu/data2/basedata/ampSLM_YH_squared_train6464'
# train_noise_6464_path = '/media/jnu/data1/data_minist_rice/cba/'
# train_noise_6464_path = '/media/jnu/data1/data_minist_rice/mnst/'
# train_noise_6464_path = '/home/jnu/data/datas/data_cut/YH_train6464_tm_to64_1/'

test_clear_6464_path = '/media/jnu/data2/basedata/ampSLM_XH_test_6464'
test_noise_6464_path = '/media/jnu/data2/basedata/ampSLM_YH_squared_test6464'


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
    return clear_imgs_train_data,noise_imgs_train_data

trian_num =7500
val_endnum =8000
def load_val_data():
    val_clear = []
    val_noise = []
    # for idx in range((index-1)*batchsize+1, index*batchsize+1):
    for idx in range(trian_num+1, val_endnum+1):
        # get the match imgs of clears and noises
        c_path = ('%s/%d.png' % (val_clear_6464_path, idx))
        n_path = ('%s/%d.png' % (val_noise_6464_path, idx))
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

    return val_clear,val_noise



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
test_minist_clear_6464_path_plus = '/media/jnu/data1/TM_test/net/testnoise1/'
test_minist_noise_6464_path_plus = '/media/jnu/data1/TM_test/net/testclear1/'


def load_minist_test_data_plus():

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
            c_path = ('%s/%d.png' % (test_minist_clear_6464_path_plus, idx))
            n_path = ('%s/%d.png' % (test_minist_noise_6464_path_plus, idx))
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

test_matfocus_clear_6464_path_plus = '/media/jnu/data1/TM_test/net/GT'  #正向
test_matfocus_noise_6464_path_plus = '/media/jnu/data1/TM_test/net/LQ'

def load_matfocus_test_data_plus():

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