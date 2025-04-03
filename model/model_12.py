import torch
import torch.nn as nn
class custom_activatipon(nn.Module):
    def __init__(self,alpha=2):
        super(custom_activatipon, self).__init__()
        self.alpha = alpha
    def forward(self,x):
        return 1 / (1+torch.exp(-self.alpha*(x)))

class sin_activation(nn.Module):
    def __init__(self,alpha=5):
        super(sin_activation, self).__init__()
        # self.alpha = alpha
    def forward(self,x):
        return torch.sin(x)

class Linear_set1(nn.Module):
    def __init__(self):
            super(Linear_set1, self).__init__() #下采样
            # self.conv = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0)
            # self.upsample7 = nn.ConvTranspose2d(4, 1, 3, stride=2, padding=1, output_padding=1)
            self.Linear1 = nn.Linear(256 * 256, 64 * 64)
            # self.Linear2 = nn.Linear(80 * 80, 64 * 64)
            # self.Linear3 = nn.Linear(64 * 64, 64 * 64)
            self.dropout = nn.Dropout(p=0.1)
            self.relu = nn.ReLU()
            self.LeakyReLU = nn.LeakyReLU(0.002)
            self.hardtanh = nn.Hardtanh(min_val=-1, max_val=1, inplace=False)  # 定义 Hardtanh 层
            self.Hardsigmoid = nn.Hardsigmoid()  # 定义 Hardtanh 层
            self.Sigmoid = nn.Sigmoid()  # 定义 Hardtanh 层
            self.Tanh = nn.Tanh()  # 定义 Hardtanh 层
            self.activation = custom_activatipon()
            self.sin_activation = sin_activation()
    def forward(self, x):
        # 下采样
        # x = x.reshape(x.shape[0], 1600, 1, 1)
        # x = self.cov0(x)
        # # x = self.upsample1(x)
        # x = self.upsample2(x)
        # x = self.upsample3(x)
        # x = self.upsample4(x)
        # x = self.upsample5(x)
        # x = self.upsample6(x)
        # x = self.upsample7(x)
        # x = self.conv(x)
        output = x.reshape(x.shape[0], -1)
        output = self.Linear1(output)
        # output = self.Linear2(output)
        # output = self.Sigmoid(output)
        # output = self.Linear2(output)
        # output = self.Linear2(output)
        output = self.Sigmoid(output)
        # output = self.Linear3(output)
        # # output = self.Linear2(output)
        # output = self.activation(output)
        # output = self.activation(output)
        output = output.reshape(x.shape[0], 64, 64).unsqueeze(1)  # 4*1*40*40
        # 全连接分块
        return output



class Linear_set1_sigmoid(nn.Module):
    def __init__(self):
            super(Linear_set1_sigmoid, self).__init__() #下采样
            # self.conv = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0)
            # self.upsample7 = nn.ConvTranspose2d(4, 1, 3, stride=2, padding=1, output_padding=1)
            self.Linear1 = nn.Linear(256 * 256, 64 * 64)
            # self.Linear2 = nn.Linear(80 * 80, 64 * 64)
            # self.Linear3 = nn.Linear(64 * 64, 64 * 64)
            self.dropout = nn.Dropout(p=0.1)
            self.relu = nn.ReLU()
            self.LeakyReLU = nn.LeakyReLU(0.002)
            self.hardtanh = nn.Hardtanh(min_val=-1, max_val=1, inplace=False)  # 定义 Hardtanh 层
            self.Hardsigmoid = nn.Hardsigmoid()  # 定义 Hardtanh 层
            self.Sigmoid = nn.Sigmoid()  # 定义 Hardtanh 层
            self.Tanh = nn.Tanh()  # 定义 Hardtanh 层
            self.activation = custom_activatipon()
            self.sin_activation = sin_activation()
    def forward(self, x):
        # 下采样
        # x = x.reshape(x.shape[0], 1600, 1, 1)
        # x = self.cov0(x)
        # # x = self.upsample1(x)
        # x = self.upsample2(x)
        # x = self.upsample3(x)
        # x = self.upsample4(x)
        # x = self.upsample5(x)
        # x = self.upsample6(x)
        # x = self.upsample7(x)
        # x = self.conv(x)
        output = x.reshape(x.shape[0], -1)
        output = self.Linear1(output)
        # output = self.Linear2(output)
        # output = self.Sigmoid(output)
        # output = self.Linear2(output)
        # output = self.Linear2(output)
        output = self.Sigmoid(output)
        # output = self.Linear3(output)
        # # output = self.Linear2(output)
        # output = self.activation(output)
        # output = self.activation(output)
        output = output.reshape(x.shape[0], 64, 64).unsqueeze(1)  # 4*1*40*40
        # 全连接分块
        return output


class Linear_set1_sigmoid_act(nn.Module):
    def __init__(self):
            super(Linear_set1_sigmoid_act, self).__init__() #下采样
            # self.conv = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0)
            # self.upsample7 = nn.ConvTranspose2d(4, 1, 3, stride=2, padding=1, output_padding=1)
            self.Linear1 = nn.Linear(256 * 256, 64 * 64)
            # self.Linear2 = nn.Linear(80 * 80, 64 * 64)
            # self.Linear3 = nn.Linear(64 * 64, 64 * 64)
            self.dropout = nn.Dropout(p=0.1)
            self.relu = nn.ReLU()
            self.LeakyReLU = nn.LeakyReLU(0.002)
            self.hardtanh = nn.Hardtanh(min_val=-1, max_val=1, inplace=False)  # 定义 Hardtanh 层
            self.Hardsigmoid = nn.Hardsigmoid()  # 定义 Hardtanh 层
            self.Sigmoid = nn.Sigmoid()  # 定义 Hardtanh 层
            self.Tanh = nn.Tanh()  # 定义 Hardtanh 层
            self.activation = custom_activatipon()
            self.sin_activation = sin_activation()
    def forward(self, x):
        # 下采样
        # x = x.reshape(x.shape[0], 1600, 1, 1)
        # x = self.cov0(x)
        # # x = self.upsample1(x)
        # x = self.upsample2(x)
        # x = self.upsample3(x)
        # x = self.upsample4(x)
        # x = self.upsample5(x)
        # x = self.upsample6(x)
        # x = self.upsample7(x)
        # x = self.conv(x)
        output = x.reshape(x.shape[0], -1)
        output = self.Linear1(output)
        # output = self.Linear2(output)
        # output = self.Sigmoid(output)
        # output = self.Linear2(output)
        # output = self.Linear2(output)
        # output = self.Sigmoid(output)
        # output = self.Linear3(output)
        # # output = self.Linear2(output)
        output = self.activation(output)
        # output = self.activation(output)
        output = output.reshape(x.shape[0], 64, 64).unsqueeze(1)  # 4*1*40*40
        # 全连接分块
        return output




class Linear_set1_sigmoid_act5(nn.Module):
    def __init__(self):
            super(Linear_set1_sigmoid_act5, self).__init__() #下采样
            # self.conv = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0)
            # self.upsample7 = nn.ConvTranspose2d(4, 1, 3, stride=2, padding=1, output_padding=1)
            self.Linear1 = nn.Linear(256 * 256, 64 * 64)
            # self.Linear2 = nn.Linear(80 * 80, 64 * 64)
            # self.Linear3 = nn.Linear(64 * 64, 64 * 64)
            self.dropout = nn.Dropout(p=0.1)
            self.relu = nn.ReLU()
            self.LeakyReLU = nn.LeakyReLU(0.002)
            self.hardtanh = nn.Hardtanh(min_val=-1, max_val=1, inplace=False)  # 定义 Hardtanh 层
            self.Hardsigmoid = nn.Hardsigmoid()  # 定义 Hardtanh 层
            self.Sigmoid = nn.Sigmoid()  # 定义 Hardtanh 层
            self.Tanh = nn.Tanh()  # 定义 Hardtanh 层
            self.activation = custom_activatipon()
            self.sin_activation = sin_activation()
    def forward(self, x):
        # 下采样
        # x = x.reshape(x.shape[0], 1600, 1, 1)
        # x = self.cov0(x)
        # # x = self.upsample1(x)
        # x = self.upsample2(x)
        # x = self.upsample3(x)
        # x = self.upsample4(x)
        # x = self.upsample5(x)
        # x = self.upsample6(x)
        # x = self.upsample7(x)
        # x = self.conv(x)
        output = x.reshape(x.shape[0], -1)
        output = self.Linear1(output)
        # output = self.Linear2(output)
        # output = self.Sigmoid(output)
        # output = self.Linear2(output)
        # output = self.Linear2(output)
        # output = self.Sigmoid(output)
        # output = self.Linear3(output)
        # # output = self.Linear2(output)
        output = self.activation(output)
        # output = self.activation(output)
        output = output.reshape(x.shape[0], 64, 64).unsqueeze(1)  # 4*1*40*40
        # 全连接分块
        return output



class Linear_set1_sigmoid_act50_plus(nn.Module):
    def __init__(self):
            super(Linear_set1_sigmoid_act50_plus, self).__init__() #下采样
            # self.conv = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0)
            # self.upsample7 = nn.ConvTranspose2d(4, 1, 3, stride=2, padding=1, output_padding=1)
            self.Linear1 = nn.Linear(256 * 256, 64 * 64)
            # self.Linear2 = nn.Linear(80 * 80, 64 * 64)
            # self.Linear3 = nn.Linear(64 * 64, 64 * 64)
            self.dropout = nn.Dropout(p=0.1)
            self.relu = nn.ReLU()
            self.LeakyReLU = nn.LeakyReLU(0.002)
            self.hardtanh = nn.Hardtanh(min_val=-1, max_val=1, inplace=False)  # 定义 Hardtanh 层
            self.Hardsigmoid = nn.Hardsigmoid()  # 定义 Hardtanh 层
            self.Sigmoid = nn.Sigmoid()  # 定义 Hardtanh 层
            self.Tanh = nn.Tanh()  # 定义 Hardtanh 层
            self.activation = custom_activatipon()
            self.sin_activation = sin_activation()
    def forward(self, x):
        # 下采样
        # x = x.reshape(x.shape[0], 1600, 1, 1)
        # x = self.cov0(x)
        # # x = self.upsample1(x)
        # x = self.upsample2(x)
        # x = self.upsample3(x)
        # x = self.upsample4(x)
        # x = self.upsample5(x)
        # x = self.upsample6(x)
        # x = self.upsample7(x)
        # x = self.conv(x)
        output = x.reshape(x.shape[0], -1)
        output = self.Linear1(output)
        # output = self.Linear2(output)
        # output = self.Sigmoid(output)
        # output = self.Linear2(output)
        # output = self.Linear2(output)
        output = self.Sigmoid(output)
        # output = self.Linear3(output)
        # # output = self.Linear2(output)
        output = self.activation(output)
        # output = self.activation(output)
        output = output.reshape(x.shape[0], 64, 64).unsqueeze(1)  # 4*1*40*40
        # 全连接分块
        return output



class Linear_set1_sigmoid_act2_MSE_L1(nn.Module):
    def __init__(self):
            super(Linear_set1_sigmoid_act2_MSE_L1, self).__init__() #下采样
            # self.conv = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0)
            # self.upsample7 = nn.ConvTranspose2d(4, 1, 3, stride=2, padding=1, output_padding=1)
            self.Linear1 = nn.Linear(256 * 256, 64 * 64)
            # self.Linear2 = nn.Linear(80 * 80, 64 * 64)
            # self.Linear3 = nn.Linear(64 * 64, 64  * 64)
            self.dropout = nn.Dropout(p=0.1)
            self.relu = nn.ReLU()
            self.LeakyReLU = nn.LeakyReLU(0.002)
            self.hardtanh = nn.Hardtanh(min_val=-1, max_val=1, inplace=False)  # 定义 Hardtanh 层
            self.Hardsigmoid = nn.Hardsigmoid()  # 定义 Hardtanh 层
            self.Sigmoid = nn.Sigmoid()  # 定义 Hardtanh 层
            self.Tanh = nn.Tanh()  # 定义 Hardtanh 层
            self.activation = custom_activatipon()
            self.sin_activation = sin_activation()
    def forward(self, x):
        output = x.reshape(x.shape[0], -1)
        output = self.Linear1(output)
        output = self.Sigmoid(output)
        output = output.reshape(x.shape[0], 64, 64).unsqueeze(1) 
        # 全连接分块
        return output