import torch
import torch.nn as nn
class custom_activatipon(nn.Module):
    def __init__(self,alpha=5):
        super(custom_activatipon, self).__init__()
        self.alpha = alpha
    def forward(self,x):
        return 1 / (1+torch.exp(-self.alpha*(x-0.8)))



class Linear_9_20(nn.Module):
    def __init__(self):
            super(Linear_9_20, self).__init__() #下采样
            # self.conv = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0)
            # self.upsample7 = nn.ConvTranspose2d(4, 1, 3, stride=2, padding=1, output_padding=1)
            self.Linear1 = nn.Linear(256 * 256, 80 * 80)
            self.Linear2 = nn.Linear(80 * 80, 64 * 64)
            self.Linear3 = nn.Linear(64 * 64, 64 * 64)
            self.dropout = nn.Dropout(p=0.1)
            self.relu = nn.ReLU()
            self.LeakyReLU = nn.LeakyReLU(0.002)
            self.hardtanh = nn.Hardtanh(min_val=-1, max_val=1, inplace=False)  # 定义 Hardtanh 层
            self.Hardsigmoid = nn.Hardsigmoid()  # 定义 Hardtanh 层
            self.Sigmoid = nn.Sigmoid()  # 定义 Hardtanh 层
            self.Tanh = nn.Tanh()  # 定义 Hardtanh 层
            self.activation = custom_activatipon()

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
        output = self.Tanh(output)
        output = self.Linear2(output)
        # output = self.Linear2(output)
        output = self.Tanh(output)
        output = self.Linear3(output)
        # output = self.Linear2(output)
        output = self.Sigmoid(output)
        # output = self.activation(output)
        output = output.reshape(x.shape[0], 64, 64).unsqueeze(1)  # 4*1*40*40
        # 全连接分块
        return output



class Linear_3_11_net_A(nn.Module):
    def __init__(self):
            super(Linear_3_11_net_A, self).__init__() #下采样
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
        # # output = self.Linear2(output)
        # output = self.Tanh(output)
        # output = self.Linear2(output)
        # # output = self.Linear2(output)
        # output = self.Tanh(output)
        # output = self.Linear3(output)
        # # output = self.Linear2(output)
        output = self.Sigmoid(output)
        # output = self.activation(output)
        output = output.reshape(x.shape[0], 64, 64).unsqueeze(1)  # 4*1*40*40
        # 全连接分块
        return output


class Linear_mutimode_net_A(nn.Module):
    def __init__(self):
            super(Linear_mutimode_net_A, self).__init__() #下采样
            # self.conv = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0)
            # self.upsample7 = nn.ConvTranspose2d(4, 1, 3, stride=2, padding=1, output_padding=1)
            self.Linear1 = nn.Linear(120 * 120, 92 * 92)
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
        # # output = self.Linear2(output)
        # output = self.Tanh(output)
        # output = self.Linear2(output)
        # # output = self.Linear2(output)
        # output = self.Tanh(output)
        # output = self.Linear3(output)
        # # output = self.Linear2(output)
        output = self.Sigmoid(output)
        # output = self.activation(output)
        output = output.reshape(x.shape[0], 92, 92).unsqueeze(1)  # 4*1*40*40
        # 全连接分块
        return output



class Linear_three_layer_net_A(nn.Module):
    def __init__(self):
            super(Linear_three_layer_net_A, self).__init__() #下采样
            # self.conv = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0)
            # self.upsample7 = nn.ConvTranspose2d(4, 1, 3, stride=2, padding=1, output_padding=1)
            self.Linear1 = nn.Linear(256 * 256, 80 * 80)
            self.Linear2 = nn.Linear(80 * 80, 64 * 64)
            # self.Linear3 = nn.Linear(64 * 64, 64 * 64)
            self.dropout = nn.Dropout(p=0.1)
            self.relu = nn.ReLU()
            self.LeakyReLU = nn.LeakyReLU(0.002)
            self.hardtanh = nn.Hardtanh(min_val=-1, max_val=1, inplace=False)  # 定义 Hardtanh 层
            self.Hardsigmoid = nn.Hardsigmoid()  # 定义 Hardtanh 层
            self.Sigmoid = nn.Sigmoid()  # 定义 Hardtanh 层
            self.Tanh = nn.Tanh()  # 定义 Hardtanh 层
            self.activation = custom_activatipon()

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
        # # output = self.Linear2(output)
        # output = self.Tanh(output)
        output = self.Linear2(output)
        # # output = self.Linear2(output)
        # output = self.Tanh(output)
        # output = self.Linear3(output)
        # # output = self.Linear2(output)
        output = self.Sigmoid(output)
        # output = self.activation(output)
        output = output.reshape(x.shape[0], 64, 64).unsqueeze(1)  # 4*1*40*40
        # 全连接分块
        return output


class Linear_3_11_net_A_RELU(nn.Module):
    def __init__(self):
            super(Linear_3_11_net_A_RELU, self).__init__() #下采样
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
        # # output = self.Linear2(output)
        # output = self.Tanh(output)
        # output = self.Linear2(output)
        # # output = self.Linear2(output)
        # output = self.Tanh(output)
        # output = self.Linear3(output)
        # # output = self.Linear2(output)
        output = self.relu(output)
        # output = self.activation(output)
        output = output.reshape(x.shape[0], 64, 64).unsqueeze(1)  # 4*1*40*40
        # 全连接分块
        return output


class Linear_3_11_net_A_LeakyReLU(nn.Module):
    def __init__(self):
            super(Linear_3_11_net_A_LeakyReLU, self).__init__() #下采样
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
        # # output = self.Linear2(output)
        # output = self.Tanh(output)
        # output = self.Linear2(output)
        # # output = self.Linear2(output)
        # output = self.Tanh(output)
        # output = self.Linear3(output)
        # # output = self.Linear2(output)
        output = self.LeakyReLU(output)
        # output = self.activation(output)
        output = output.reshape(x.shape[0], 64, 64).unsqueeze(1)  # 4*1*40*40
        # 全连接分块
        return output


class Linear_3_11_net_B(nn.Module):
    def __init__(self):
            super(Linear_3_11_net_B, self).__init__() #下采样
            # self.conv = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0)
            # self.upsample7 = nn.ConvTranspose2d(4, 1, 3, stride=2, padding=1, output_padding=1)
            self.Linear1 = nn.Linear(64 * 64,256 * 256)
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

    def forward(self, x):
        output = x.reshape(x.shape[0], -1)
        output = self.Linear1(output)
        output = self.Sigmoid(output)
        output = output.reshape(x.shape[0], 256, 256).unsqueeze(1) 
        # 全连接分块
        return output

class Linear_mutimode_net_B(nn.Module):
    def __init__(self):
            super(Linear_mutimode_net_B, self).__init__() #下采样
            # self.conv = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0)
            # self.upsample7 = nn.ConvTranspose2d(4, 1, 3, stride=2, padding=1, output_padding=1)
            self.Linear1 = nn.Linear(92 * 92,120 * 120)
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
        # # output = self.Linear2(output)
        # output = self.Tanh(output)
        # output = self.Linear2(output)
        # # output = self.Linear2(output)
        # output = self.Tanh(output)
        # output = self.Linear3(output)
        # # output = self.Linear2(output)
        output = self.Sigmoid(output)
        # output = self.activation(output)
        output = output.reshape(x.shape[0], 120, 120).unsqueeze(1)  # 4*1*40*40
        # 全连接分块
        return output


class Linear_three_layer_net_B(nn.Module):
    def __init__(self):
            super(Linear_three_layer_net_B, self).__init__() #下采样
            # self.conv = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0)
            # self.upsample7 = nn.ConvTranspose2d(4, 1, 3, stride=2, padding=1, output_padding=1)
            self.Linear1 = nn.Linear(64 * 64,80 * 80)
            self.Linear2 = nn.Linear(80 * 80, 256 * 256)
            # self.Linear3 = nn.Linear(64 * 64, 64 * 64)
            self.dropout = nn.Dropout(p=0.1)
            self.relu = nn.ReLU()
            self.LeakyReLU = nn.LeakyReLU(0.002)
            self.hardtanh = nn.Hardtanh(min_val=-1, max_val=1, inplace=False)  # 定义 Hardtanh 层
            self.Hardsigmoid = nn.Hardsigmoid()  # 定义 Hardtanh 层
            self.Sigmoid = nn.Sigmoid()  # 定义 Hardtanh 层
            self.Tanh = nn.Tanh()  # 定义 Hardtanh 层
            self.activation = custom_activatipon()

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
        # # output = self.Linear2(output)
        # output = self.Tanh(output)
        output = self.Linear2(output)
        # # output = self.Linear2(output)
        # output = self.Tanh(output)
        # output = self.Linear3(output)
        # # output = self.Linear2(output)
        output = self.Sigmoid(output)
        # output = self.activation(output)
        output = output.reshape(x.shape[0], 256, 256).unsqueeze(1)  # 4*1*40*40
        # 全连接分块
        return output

class Linear_3_11_net_B_RElU(nn.Module):
    def __init__(self):
            super(Linear_3_11_net_B_RElU, self).__init__() #下采样
            # self.conv = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0)
            # self.upsample7 = nn.ConvTranspose2d(4, 1, 3, stride=2, padding=1, output_padding=1)
            self.Linear1 = nn.Linear(64 * 64,256 * 256)
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
        # # output = self.Linear2(output)
        # output = self.Tanh(output)
        # output = self.Linear2(output)
        # # output = self.Linear2(output)
        # output = self.Tanh(output)
        # output = self.Linear3(output)
        # # output = self.Linear2(output)
        output = self.relu(output)
        # output = self.activation(output)
        output = output.reshape(x.shape[0], 256, 256).unsqueeze(1)  # 4*1*40*40
        # 全连接分块
        return output

class Linear_3_11_net_B_no_sigmoid(nn.Module):
    def __init__(self):
            super(Linear_3_11_net_B_no_sigmoid, self).__init__() #下采样
            # self.conv = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0)
            # self.upsample7 = nn.ConvTranspose2d(4, 1, 3, stride=2, padding=1, output_padding=1)
            self.Linear1 = nn.Linear(64 * 64,256 * 256)
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
        # # output = self.Linear2(output)
        # output = self.Tanh(output)
        # output = self.Linear2(output)
        # # output = self.Linear2(output)
        # output = self.Tanh(output)
        # output = self.Linear3(output)
        # # output = self.Linear2(output)
        # output = self.Sigmoid(output)
        # output = self.activation(output)
        output = output.reshape(x.shape[0], 256, 256).unsqueeze(1)  # 4*1*40*40
        # 全连接分块
        return output


