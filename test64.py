import data_solve
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
from data_solve import data_pre_treated6464 ,data_pre_treated4040, data_save,optimizer,data_save4040



path = '/media/jnu/data2/model_3_11/Net-A/model_801.pth'
#
# path = "/media/jnu/data2/model_3_12/Linear_set1_sigmoid_act2_MSE_nmse_plus_plus_plus_P/model_2001.pth"
path_result = '/home/jnu/视频/yuce'

if not os.path.exists(path_result):
     os.makedirs(path_result)

batch_size = 6
# 将模型放到GPU上
device_ids = [0]  # 指定使用的GPU设备ID
device = torch.device("cuda:%d" % device_ids[0] if torch.cuda.is_available() else "cpu")



if __name__ == '__main__':
    # Linear_net6464_c = torch.load(path, map_location=torch.device('CPU'))
    conv_Linear_net3 = torch.load(path)
    # state_dict = torch.load(path)
    # new_state_dict = {}
    # for key in state_dict.keys():
    #     new_key = key.replace("module.", "")
    #     new_state_dict[new_key] = state_dict[key]
    # model.load_state_dict(state_dict,)

    # model.to(device)
    conv_Linear_net3.eval()
    with torch.no_grad():
            # 将数据转换为 PyTorch 张量
            # clear_test_tensor, noise_test_tensor = data_pre_treated6464.load_test_data()
            # # 创建 TensorDataset 对象
            # test_dataset = TensorDataset(noise_test_tensor, clear_test_tensor)
            # # 创建 DataLoader 对象
            # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # 加载数据
            # data_save4040.datasavetestminist6464_plus(model=conv_Linear_net3, loader=test_loader, device=device,
            #                                           path=path_result, epoch=1)
            #
            clear_test_tensor, noise_test_tensor = data_pre_treated6464.load_test_data()
            # 创建 TensorDataset 对象
            test_dataset = TensorDataset(noise_test_tensor,clear_test_tensor)
            # 创建 DataLoader 对象
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # 加载数据
            data_save4040.datasavetestminist6464_plus(model=conv_Linear_net3, loader=test_loader, device=device, path=path_result, epoch=1)

















