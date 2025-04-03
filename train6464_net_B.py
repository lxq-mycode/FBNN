import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import model6464to256256
from model import model_9_20
from data_solve import data_pre_treated6464, data_save, optimizer, data_save4040
import matplotlib.pyplot as plt
from model import ssim
import random
from torchsummary import summary
import time
# 初始化设置
# model_flag = data_pre_treated.model_flag #如果等于1就是1616  2 4040 3 6464
model_flag = 3
batch_size = 4  # 训练批次
num_epochs = 2000  # 迭代次数
trian_num = 49152
CEloss = 0
TVloss = 0

# 将模型放到多个GPU上
device_ids = [0]  # 指定使用的GPU设备ID
device = torch.device("cuda:%d" % device_ids[0] if torch.cuda.is_available() else "cpu")

# 模型和路径选择
if model_flag == 3:
    model = model_9_20.Linear_3_11_net_B()
    path = '/media/jnu/data2/model_3_11/Net-B/'
if not os.path.exists(path):
    os.makedirs(path)

model = torch.nn.DataParallel(model, device_ids=device_ids)
print(summary(model, (1, 64, 64)))
model.to(device)
criterion1 = nn.MSELoss()
criterion2 = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0001)
# 优化器设置
optimizer_set = optim.Adam(model.parameters(), lr=0.00001)


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


def train(model, optimizer, device, train_loader):
    # 读取TensorDataset以及创建DataLoader 对象
    train_loss = 0
    model.train()
    for i, (input, target) in enumerate(train_loader):
        # rotation_angle = torch.randint(low=1,high=5,size=(1,)).item()*90
        # noise_imgs_train_tensor = torch.rot90(noise_imgs_train_tensor,k=rotation_angle//90,dims=(2,3))
        # clear_imgs_train_tensor = torch.rot90(clear_imgs_train_tensor,k=rotation_angle//90,dims=(2,3))
        input, target = input.to(device), target.to(device)
        input = input.type(torch.cuda.FloatTensor)
        output = model(input)
        target = target.type(torch.cuda.FloatTensor)
        loss = criterion1(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    return train_loss


def val(model, criterion, device, val_loader):
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            # rotation_angle = torch.randint(low=1,high=5,size=(1,)).item()*90
            # input = torch.rot90(input,k=rotation_angle//90,dims=(2,3))
            # target = torch.rot90(target,k=rotation_angle//90,dims=(2,3))
            input, target = input.to(device), target.to(device)
            input = input.type(torch.cuda.FloatTensor)
            # input = input - 0.07339818
            output = model(input)
            # for i in range(output.shape[0]):
            #     for j in range(16):
            #         for k in range(16):
            #             if output[i,:,j,k]>0.5:
            #                 output[i,:,j,k]=1
            #             else:
            #                 output[i, :, j, k] = 0
            target = target.type(torch.cuda.FloatTensor)
            # loss = ssim.NMSELoss()(output, target)
            loss = criterion1(output, target)
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
    index = 0
    index1 = 0

    # 初始化训练时间记录变量
    start_time = time.time()  # 记录训练开始时间

    # clear_val_tensor, noise_val_tensor = data_pre_treated6464.load_val_data()
    # val_dataset = TensorDataset(noise_val_tensor, clear_val_tensor)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)  # 加载数据

    clear_test_tensor, noise_test_tensor = data_pre_treated6464.load_test_data()
    test_dataset = TensorDataset(clear_test_tensor,noise_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # 加载数据
    lst = list(range(1, trian_num + 1))
    lst1 = random.sample(lst, 1000)
    clear_train_tensor, noise_train_tensor = data_pre_treated6464.load_train_data(lst1)
    train_dataset = TensorDataset(clear_train_tensor,noise_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # 加载数据
    for epoch in range(1, num_epochs + 1):
        # 每200次迭代记录一次训练时间
        if epoch % 200== 0:
            end_time = time.time()  # 记录当前时间
            elapsed_time = end_time - start_time  # 计算训练耗时

            # 将训练时间转换为时分秒格式
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f"Epoch {epoch}: Training time elapsed: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds")

            start_time = time.time()  # 重置起始时间

        # 训练过程

        lst1 = random.sample(lst, 1000)
        clear_train_tensor, noise_train_tensor = data_pre_treated6464.load_train_data(lst1)
        train_dataset = TensorDataset(clear_train_tensor,noise_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # 加载数据
        train_loss = train(model=model, optimizer=optimizer_set, device=device, train_loader=train_loader)
        print('Epoch [{}/{}], Train Loss: {:.4f}'.format(epoch, num_epochs, train_loss))
        train_losses.append(train_loss)
        val_loss = val(model=model, criterion=criterion1, device=device, val_loader=test_loader)
        val_losses.append(val_loss)

        print('Epoch: {} test Loss: {:.4f}'.format(epoch, val_loss))
        if epoch % 200 == 0 and epoch > 0:
            # 训练集
            data_save.datasavetrain(model=model, loader=train_loader, device=device, path=path, epoch=epoch)
            # 验证集
            data_save.datasaveval(model=model, loader=test_loader, device=device, path=path, epoch=epoch)
            # 测试集

        if epoch % 200 == 0 and epoch > 0:
            torch.save(model, path + 'model_{}.pth'.format(epoch + 1))
            # 绘制loss曲线
        if epoch % 200 == 0 and epoch > 0:
            plt.plot(range(1, epoch + 1), train_losses, label='Train Loss')
            plt.plot(range(1, epoch + 1), val_losses, label='Test Loss')
            plt.xlabel('Epoch', fontsize=14)  # 设置x轴标签字体大小
            plt.ylabel('Loss', fontsize=14)  # 设置y轴标签字体大小
            plt.legend(fontsize=14)  # 设置图例字体大小
            # plt.title('Training and Validation Loss', fontsize=16)  # 设置标题字体大小
            plt.savefig(path + 'loss%d.png' % epoch)  # 文件名可以自定义
            plt.show()

    # 训练结束后记录总时间
    total_time = time.time() - start_time
    # 将总时间转换为时分秒格式
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total training time: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds")


