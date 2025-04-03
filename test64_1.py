import data_solve
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
from data_solve import data_pre_treated ,data_pre_treated6464, data_save,optimizer,data_save4040

#初始化设置
model_flag = 2 #如果等于1就是1616  2 4040 3 6464
# 路径选择
# if model_flag == 1:
#     # path = '/home/jnu/data/datas1616/My_net1616_V2/model_21.pth'
#     path = '/media/jnu/data2/model6464to256256/Linear_conv/model_221.pth'
#     path_result = '/media/jnu/data2/test_result/Linear_conv6464/'
# elif model_flag == 2:
    # path = '/media/jnu/data2/model_test/Linear_net4040_c_plus_resultmse2/model_136.pth'
# path = '/media/jnu/data2/model6464to256256/Linear_conv_GT_LQderectly/model_41.pth'
# path_result = '/media/jnu/data2/foucs_image/LQ/'


# path = '/media/jnu/data2/model_Linear/Linear_net_6464_one_linear/model_101.pth'
# path = '/media/jnu/data2/model6464to256256/Linear_two/model_121.pth'
# path_result = '/media/jnu/data1/focus_test/net/'
# path = '/media/jnu/data2/model_new/conv_Linear_net36464/model_41.pth'
# path = '/media/jnu/data2/model6464to256256/Linear_conv/model_81.pth'
# path_result = '/media/jnu/data1/now_shiyan/shabannet/'
# path = '/media/jnu/data2/model6464to256256/Linear_conv_GT_LQderectly2000/model_41.pth'
# path = '/media/jnu/data1/model/bin_model20000/model_41.pth'
# path = '/media/jnu/data1/model/for20000_bin_model20000/model_21.pth'

# path = "/media/jnu/data2/model_12_TM/L1loss_Linear_3L_new/model_21.pth"
# path = '/media/jnu/data2/model_Linear/Linear_net_6464_one_linear/model_61.pth'
# path = '/media/jnu/data2/model6464to256256/Linear_conv_GT_LQderectly5000/model_21.pth'
# path = '/media/jnu/data2/model6464to256256/Linear_conv_GT_LQderectly/model_41.pth'
# path = '/media/jnu/data2/model_9_9/Linear_3L_1/model_11.pth'
# path = "/media/jnu/data2/model_9_9/Linear_3L_true/model_401.pth"
# path = "/media/jnu/data1/model/bin_model20000/model_41.pth"
# path = "/media/jnu/data2/model_9_27/Linear_3L_true/model_31.pth"

# path = "/media/jnu/data1/model/for20000_bin_model20000/model_21.pth"
# path = "/media/jnu/data2/model_12_TM/L1loss_Linear_3L_new/model_31.pth"
# path = '/media/jnu/data2/model_9_27/Linear_3L_true/model_31.pth'
path = '/media/jnu/data2/model_3_11/Net-B/model_401.pth'
path_result = '/home/jnu/视频/SP'
# path_result = '/media/jnu/data2/cyclenet_test_result/Linear_conv6464/minist/'

if not os.path.exists(path_result):
     os.makedirs(path_result)

batch_size = 4
# 将模型放到GPU上
device_ids = [0]  # 指定使用的GPU设备ID
device = torch.device("cuda:%d" % device_ids[0] if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    # Linear_net6464_c = torch.load(path, map_location=torch.device('CPU'))
    Linear_conv_1 = torch.load(path)
    # state_dict = torch.load(path)
    # new_state_dict = {}
    # for key in state_dict.keys():
    #     new_key = key.replace("module.", "")
    #     new_state_dict[new_key] = state_dict[key]
    # model.load_state_dict(state_dict,)

    # model.to(device)
    Linear_conv_1.eval()
    with torch.no_grad():
            # 将数据转换为 PyTorch 张量
            clear_test_tensor, noise_test_tensor = data_pre_treated6464.load_minist_test_data_plus()
            # 创建 TensorDataset 对象
            test_dataset = TensorDataset(clear_test_tensor,noise_test_tensor)
            # 创建 DataLoader 对象
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # 加载数据
            data_save4040.datasavetestminist6464_plus(model=Linear_conv_1, loader=test_loader, device=device, path=path_result, epoch=1)















