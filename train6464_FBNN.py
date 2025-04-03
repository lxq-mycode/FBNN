import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import model_12
from model import model_Linear
from data_solve import data_pre_treated ,data_pre_treated6464, data_save,optimizer,data_save4040
import matplotlib.pyplot as plt
from model import ssim
import itertools
import random
import time
from torchsummary import summary
#初始化设置
batch_size = 4  #训练批次
num_epochs = 2000 #迭代次数
traindata_num = 49152

CEloss =0
TVloss =0
# 将模型放到多个GPU上
device_ids = [0]  # 指定使用的GPU设备ID
device = torch.device("cuda:%d" % device_ids[0] if torch.cuda.is_available() else "cpu")
# 模型和路径选择
# modeldevice_ids = [0]  # 指定使用的GPU设备ID
# modeldevice = torch.device("cuda:%d" % device_ids[0] if torch.cuda.is_available() else "cpu")

#

# model_path = "/media/jnu/data2/model_12_TM/L1loss_Linear_3L_new/model_31.pth"


model_path = "/media/jnu/data2/model_3_11/Net-B/model_1201.pth"
# model_path = "/media/jnu/data2/model_12_TM/L1loss_Linear_3L_new/model_31.pth"


model_Linear_inver = model_12.Linear_set1_sigmoid_act2_MSE_L1()
path = '/media/jnu/data2/model_3_12/Linear_set1_sigmoid_act2_MSE_nmse_plus_plus_plus_P/'
#plus是net-A输出给net-B,然后散班做loss,然后net-B输出回头来给net-A,输出到slm做loss
#plusplus是直接中间的SLM做loss,然后，以及实际的SLM给net-B,输出然后于net-A的输出给net-B的输出做loss
#plusplusplus 是net-A的输出给net-B.然后net-B的输出和输入做loss加上中间的slm的loss,然后
# elif model_flag == 4:
#     # model = model6464.My_net6464()
#     model = model_design.TCNN_Linear_net()
#     path = '/home/jnu/data/datas2828/TCNN_Linear_net2828V1/'
if not os.path.exists(path):
    os.makedirs(path)



class NMSELoss(nn.Module):
    def __init__(self):
        super(NMSELoss, self).__init__()

    def forward(self, output, target):
        mse = torch.mean((output - target) ** 2)
        norm = torch.mean(target ** 2)
        return mse / (norm + 1e-6)  # 加上小常数以避免除零


conv_Linear_net3 = torch.load(model_path)
conv_Linear_net3.eval()
for param in conv_Linear_net3.parameters():
    param.requires_grad=False


model_Linear_inver = torch.nn.DataParallel(model_Linear_inver, device_ids=device_ids)
model_Linear_inver.to(device)
print(summary(model_Linear_inver,(1,256,256)))


criterionmse = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterionnmse = NMSELoss()
#优化器设置
optimizer_set = optim.Adam(model_Linear_inver.parameters(), lr=0.00001)
# optim_params = []
# beta1=0.9
# beta2=0.99
# for (
#         k,
#         v,
# ) in model.named_parameters():  # can optimize for a part of the model
#     # if 'NAFBlock' in k:
#     #    v.requires_grad = False
#     if v.requires_grad:
#         optim_params.append(v)
# optimizer_set = optimizer.Lion(
#     optim_params,
#     lr=0.00003,
#     weight_decay=0,
#     betas=(beta1, beta2),
# )


def train(model, optimizer, device, train_loader, conv_Linear_net3):
    model.train()
    conv_Linear_net3.eval()
    train_loss = 0
    penalty_count = 0  # 惩罚次数计数器
    lambda1 = 30
    lambda2 = 30

    # 初始化 L1_ref 和 L2_ref 为 None，用于第一次训练时记录初始值
    L1_ref, L2_ref = None, None

    for i, (input, target) in enumerate(train_loader):
        input, target = input.to(device), target.to(device)

        # 确保input和target都是Float类型
        input = input.float()
        target = target.float()

        output = model(input)
        with torch.no_grad():
            output1 = conv_Linear_net3(output)

        output1 = output1.to(device)

        # 确保loss1和loss2都是Float类型
        loss1 = criterionmse(output, target).float()
        loss2 = criterionnmse(output1, input).float()

        # 在第一次迭代时，记录loss1和loss2的初始值作为基准值
        if L1_ref is None and L2_ref is None:
            L1_ref = loss1.item()
            L2_ref = loss2.item()

        # 加入惩罚项
        P1 = torch.relu(loss1 - L1_ref)
        P2 = torch.relu(loss2 - L2_ref)

        # 统计需要施加惩罚的次数
        if P1.item() > 0:
            penalty_count += 1
        if P2.item() > 0:
            penalty_count += 1

        # 惩罚函数
        P = lambda1 * P1 + lambda2 * P2
        P = P.to(device)

        # 将惩罚项加到总损失中
        loss = loss1 + loss2+P

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新 L1_ref 和 L2_ref 为当前的 loss1 和 loss2
        L1_ref = loss1.item()
        L2_ref = loss2.item()

        train_loss += loss.item()

    # 确保train_loss的计算是基于正确的批次数量
    train_loss /= len(train_loader)

    print(f"本轮训练惩罚次数: {penalty_count}")
    return train_loss


def val(model, device,val_loader):
    val_loss = 0
    model.eval()
    with torch.no_grad():
         for i, (input, target) in enumerate(val_loader):
            # input1 = input[:, :, 0:128, 0:128]
            # input2 = input[:, :, 128:256, 0:128]
            # input3 = input[:, :, 0:128, 128:256]
            # input4 = input[:, :, 128:256, 128:256]
            # input = torch.cat((input1, input2, input3, input4), dim=1)
            input, target = input.to(device), target.to(device)
            input = input.type(torch.cuda.FloatTensor)
            output = model(input)
            # output1 = conv_Linear_net3(output)
            loss = criterionmse(output, target)
            val_loss += loss.item()
            '''
            第一个batch的时候，将模型的输入、输出和目标图像从GPU内存中转移到CPU内存中，
            并将通道维的位置从第二个位置（在PyTorch中是第三个维度）转移到最后一个位置。
            这是因为在PyTorch中，图像的维度顺序通常是(batch_size, channels, height, width)
            而在numpy中通常是(batch_size, height, width, channels)，所以需要进行维度的转换。
            这样做的目的是为了方便后续的图像保存操作，因为PIL库中读取图像时默认的维度顺序也是(batch_size, height, width, channels)。
            '''
    val_loss /= len(val_loader)
    return val_loss




if __name__ == '__main__':

    train_losses = []
    val_losses = []
    test_losses = []
    val_clear = []
    val_noise = []
    test_clear = []
    test_noise = []
    train_noise = []
    train_clear = []
    index=0
    index1=0
    lst = list(range(1, 49152))
    # clear_val_tensor, noise_val_tensor = data_pre_treated6464.load_val_data()
    # val_dataset = TensorDataset(noise_val_tensor, clear_val_tensor)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)  # 加载数据

    clear_test_tensor, noise_test_tensor = data_pre_treated6464.load_test_data()
    test_dataset = TensorDataset(noise_test_tensor, clear_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # 加载数据

    lst1 = random.sample(lst, 1000)
    clear_train_tensor, noise_train_tensor = data_pre_treated6464.load_train_data(lst1)
    train_dataset = TensorDataset(noise_train_tensor, clear_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # 加载数据
    # cyclic_lst = itertools.cycle(lst)
    # 初始化训练时间记录变量
    start_time = time.time()  # 记录训练开始时间
    for epoch in range(1,num_epochs+1):

            # 训练过程

            lst1 = random.sample(lst, 1000)
            clear_train_tensor, noise_train_tensor = data_pre_treated6464.load_train_data(lst1)
            train_dataset = TensorDataset(noise_train_tensor, clear_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # 加载数据

            train_loss = train(model=model_Linear_inver,optimizer=optimizer_set,device=device,train_loader=train_loader,conv_Linear_net3=conv_Linear_net3)
            train_losses.append(train_loss)
            print('Epoch [{}/{}], Train Loss: {:.4f}'.format(epoch , num_epochs,train_loss))
            val_loss = val(model=model_Linear_inver, device=device,val_loader=test_loader)
            val_losses.append(val_loss)
            print('Epoch: {} Validation Loss: {:.4f}'.format(epoch , val_loss))
            # test_loss = val(model=model_Linear_inver,optimizer=optimizer_set,device=device,train_loader=test_loader)
            # test_losses.append(train_loss)
            # print('Epoch [{}/{}], test Loss: {:.4f}'.format(epoch , num_epochs,test_loss))
            # 每200次迭代记录一次训练时间
            if epoch % 200 == 0:
                end_time = time.time()  # 记录当前时间
                elapsed_time = end_time - start_time  # 计算训练耗时

                # 将训练时间转换为时分秒格式
                hours, remainder = divmod(elapsed_time, 3600)
                minutes, seconds = divmod(remainder, 60)
                print(
                    f"Epoch {epoch}: Training time elapsed: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds")

                start_time = time.time()  # 重置起始时间

            if epoch % 50 == 0 and epoch > 0:
                # 保存图像
                # 训练集
                data_save4040.datasavetrain(model=model_Linear_inver, loader=train_loader, device=device, path=path, epoch=epoch)
                # 验证集
                # data_save4040.datasaveval(model=model_Linear_inver, loader=val_loader, device=device, path=path, epoch=epoch)
                # 测试集
                data_save4040.datasavetest(model=model_Linear_inver, loader=test_loader, device=device, path=path, epoch=epoch)

                # 保存模型
            if epoch % 200 == 0 and epoch > 0:
                    torch.save(model_Linear_inver , path+'model_{}.pth'.format(epoch + 1))
                    #绘制loss曲线
                    plt.plot(range(1, epoch + 1), train_losses, label='Train Loss')
                    plt.plot(range(1, epoch + 1), val_losses, label='Test Loss')
                    plt.xlabel('Epoch', fontsize=14)  # 设置x轴标签字体大小
                    plt.ylabel('Loss', fontsize=14)  # 设置y轴标签字体大小
                    plt.legend(fontsize=14)  # 设置图例字体大小
                    plt.savefig(path + 'loss%d.png' % epoch)  # 文件名可以自定义
                    # 显示图像
                    plt.show()

    # 训练结束后记录总时间
    total_time = time.time() - start_time
    # 将总时间转换为时分秒格式
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total training time: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds")















