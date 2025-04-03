import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import model6464to256256
from model import model17_TMweitiao
from data_solve import  data_pre_treated6464, data_save,optimizer,data_save4040
import matplotlib.pyplot as plt
from model import ssim
import random
from torchsummary import summary
#初始化设置
batch_size = 4  #训练批次
num_epochs = 1000 #迭代次数
traindata_num = 48000

CEloss = 0
TVloss = 0


# 模型和路径选择
# modeldevice_ids = [0]  # 指定使用的GPU设备ID
# modeldevice = torch.device("cuda:%d" % device_ids[0] if torch.cuda.is_available() else "cpu")

class custom_activatipon(nn.Module):
    def __init__(self,alpha=50):
        super(custom_activatipon, self).__init__()
        self.alpha = alpha
    def forward(self,x):
        return 1/ (1+torch.exp(-self.alpha*(x-0.5)))


class NMSELoss(nn.Module):
    def __init__(self):
        super(NMSELoss, self).__init__()

    def forward(self, output, target):
        mse = torch.mean((output - target) ** 2)
        norm = torch.mean(target ** 2)
        return mse / (norm + 1e-6)  # 加上小常数以避免除零


# model_Linear_inver = model17_TMweitiao.Linear_set1_sigmoid()
path = '/media/jnu/data2/model_3_18/L1loss_weitiao_relu/'

if not os.path.exists(path):
    os.makedirs(path)



class NMSELoss(nn.Module):
    def __init__(self):
        super(NMSELoss, self).__init__()

    def forward(self, output, target):
        mse = torch.mean((output - target) ** 2)
        norm = torch.mean(target ** 2)
        return mse / (norm + 1e-6)  # 加上小常数以避免除零

# model_path = "/media/jnu/data2/model_new/conv_Linear_net36464/model_41.pth"



# model_path_TM = "/media/jnu/data2/model_12_TM/L1loss_Linear_3L_new/model_31.pth"
model_path_TM = '/media/jnu/data2/model_3_18/Net-B_RELU/model_1001.pth'
# model_path =  "/media/jnu/data1/model/for20000_bin_model20000/model_21.pth"
conv_Linear_net3 = torch.load(model_path_TM)
conv_Linear_net3.eval()
for param in conv_Linear_net3.parameters():
    param.requires_grad=False
device_ids_premode = [0]  # 指定使用的GPU设备ID
device_premode = torch.device("cuda:%d" % device_ids_premode[0] if torch.cuda.is_available() else "cpu")



# 将模型放到多个GPU上
device_ids = [1]  # 指定使用的GPU设备ID
device = torch.device("cuda:%d" % device_ids[0] if torch.cuda.is_available() else "cpu")

model_path = '/media/jnu/data2/model_3_12/Linear_set1_sigmoid_act2_MSE_nmse_plus_plus_plus_P/model_401.pth'
model = torch.load(model_path)
model_Linear_inver = torch.nn.DataParallel(model, device_ids=device_ids)
model_Linear_inver.to(device)
print(summary(model_Linear_inver,(1,256,256)))


criterionmse = nn.L1Loss()
criterionL1 = nn.L1Loss()
#优化器设置
optimizer_set = optim.Adam(model_Linear_inver.parameters(), lr=0.000001)

def train(model, optimizer, device, train_loader, conv_Linear_net3):
    model.train()
    train_loss = 0
    for i, (input, target) in enumerate(train_loader):
        input, target = input.to(device), target.to(device)
        input = input.float()
        # target = target.float()  # 确保target是Float类型

        output1 = model(input)
        output1 = output1.to(device_premode)  # 转移到conv_Linear_net3所需的设备
        output = conv_Linear_net3(output1)  # 不需要torch.no_grad()

        output = output.to(device)
        # loss = criterionmse(output, input).float()  # 确保loss是Float类型
        # 创建权重矩阵
        # weight = torch.ones_like(output)
        # center = 128  # 256/2
        # weight[:, :, center-8:center+8, center-8:center+8] =1000  # 中间区域权重为10

        loss = criterionmse(output, input).float()  # 确保loss是Float类型

        # 加权损失
        # weighted_loss = torch.mean(loss * weight)

        optimizer.zero_grad()
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新模型参数
        train_loss += loss.item()

    train_loss /= len(train_loader)  # 计算平均损失
    return train_loss


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
    lst = list(range(1, traindata_num+1))
    # clear_val_tensor, noise_val_tensor = data_pre_treated6464.load_val_data()
    # val_dataset = TensorDataset(noise_val_tensor,clear_val_tensor)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)  # 加载数据
    clear_test_tensor, noise_test_tensor = data_pre_treated6464.load_minist_test_data_plus()
    # clear_test_tensor, noise_test_tensor = data_pre_treated6464.load_test_data()
    test_dataset = TensorDataset(noise_test_tensor , clear_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # 加载数据

    for epoch in range(1,num_epochs+1):
            if epoch % 10 == 0 or epoch == 1:
                data_save4040.datasavetest(model=model_Linear_inver, loader=test_loader, device=device, path=path,
                                           epoch=epoch)
            train_loss = train(model=model_Linear_inver,optimizer=optimizer_set,device=device,train_loader=test_loader,conv_Linear_net3=conv_Linear_net3)
            train_losses.append(train_loss)
            print('Epoch [{}/{}], Train Loss: {:.4f}'.format(epoch , num_epochs,train_loss))
                    # 保存模型
            if epoch % 200 == 0 and epoch > 0:
                    torch.save(model_Linear_inver , path+'model_{}.pth'.format(epoch + 1))
                    #绘制loss曲线
                    # plt.plot(range(1, epoch + 1), val_losses, label='Test Loss')
                    # 绘制loss曲线
                    plt.plot(range(1, epoch + 1), train_losses, label='Train Loss')
                    plt.xlabel('Epoch', fontsize=14)  # 设置x轴标签字体大小
                    plt.ylabel('Loss', fontsize=14)  # 设置y轴标签字体大小
                    plt.legend(fontsize=14)  # 设置图例字体大小
                    plt.savefig(path + 'loss%d.png' % epoch)  # 文件名可以自定义
                    # 显示图像
                    plt.show()















