import numpy as np
from PIL import Image
import os
import torch
import skimage.metrics
import torch
import torch.nn as nn
import time
class custom_activatipon(nn.Module):
    def __init__(self,alpha=50):
        super(custom_activatipon, self).__init__()
        self.alpha = alpha
    def forward(self,x):
        return 1/ (1+torch.exp(-self.alpha*(x-0.5)))


def sample_imagestrain(epoch ,batch_index, clear, noise_img2clear_img, path):
    clear = clear.cpu().numpy()
    # noise_img2clear_img = noise_img2clear_img.cpu().numpy()
    noise_img2clear_img = noise_img2clear_img.detach().cpu().numpy()

    clear = clear.squeeze(0)
    noise_img2clear_img = noise_img2clear_img.squeeze(0)
    # writer.add_image("cleartrain", clear, epoch)
    # writer.add_image("noisetain", noise, epoch)
    # writer.add_image("noise_img2clear_imgtrain", noise_img2clear_img, epoch)
    clear = np.uint8(clear*255)
    noise_img2clear_img = np.uint8(noise_img2clear_img*255)
    print('训练集第',epoch,'迭代','batch_index',batch_index,'ssim:',skimage.metrics.structural_similarity(clear,noise_img2clear_img))
    print('训练集第',epoch,'迭代','batch_index',batch_index,'psnr:',skimage.metrics.peak_signal_noise_ratio(clear,noise_img2clear_img))
    img1 = clear.flatten()
    img2 = noise_img2clear_img.flatten()
    print('训练集第', epoch, '迭代', 'batch_index', batch_index, '的相关系数是:',np.corrcoef(img1,img2)[0,1])
    clear = Image.fromarray(clear)
    noise_img2clear_img = Image.fromarray(noise_img2clear_img)

    # 保存图片
    path =path + "/trainepoch%d" % epoch
    if not os.path.exists(path):
        os.makedirs(path)
    clear.save(path + "/cleartrain%d.png" % batch_index)
    noise_img2clear_img.save(path + "/noise_img2clear_imgtrain%d.png" % batch_index)


def sample_imagesval(epoch, batch_index, clear, noise_img2clear_img, path):
    clear = clear.cpu().numpy()
    # noise_img2clear_img = noise_img2clear_img.cpu().numpy()
    noise_img2clear_img = noise_img2clear_img.detach().cpu().numpy()

    # writer.add_image("clearval", clear, epoch)
    # writer.add_image("noiseval", noise, epoch)
    # writer.add_image("noise_img2clear_imgval", noise_img2clear_img, epoch)
    clear = clear.squeeze(0)
    noise_img2clear_img = noise_img2clear_img.squeeze(0)
    clear = np.uint8(clear*255)
    noise_img2clear_img = np.uint8(noise_img2clear_img*255)
    print('验证集第',epoch,'迭代','batch_index',batch_index,'ssim:',skimage.metrics.structural_similarity(clear,noise_img2clear_img))
    print('验证集第',epoch,'迭代','batch_index',batch_index,'psnr:',skimage.metrics.peak_signal_noise_ratio(clear,noise_img2clear_img))
    img1 = clear.flatten()
    img2 = noise_img2clear_img.flatten()
    print('验证集第', epoch, '迭代', 'batch_index', batch_index, '的相关系数是:',np.corrcoef(img1,img2)[0,1])
    clear = Image.fromarray(clear)
    noise_img2clear_img = Image.fromarray(noise_img2clear_img)

    path = path + "/valepoch%d" % epoch
    if not os.path.exists(path):
        os.makedirs(path)
    clear.save(path + "/clearval%d.png" % batch_index)
    noise_img2clear_img.save(path + "/noise_img2clear_imgval%d.png" % batch_index)


def sample_imagestest(epoch, batch_epoch,batch_index, clear,noise, noise_img2clear_img, path):
    clear = clear.cpu().numpy()
    noise_img2clear_img = noise_img2clear_img.detach().cpu().numpy()
    noise =noise.detach().cpu().numpy()
    clear = clear.squeeze(0)
    noise_img2clear_img = noise_img2clear_img.squeeze(0)
    noise = noise.squeeze(0)
    clear = np.uint8(255.0 * clear)
    noise = np.uint8(255.0 * noise)
    noise_img2clear_img = np.uint8(255.0 * noise_img2clear_img)
    # noise_img2clear_img1 = np.uint8(255.0 * noise_img2clear_img1)

    print('测试集第',epoch,'迭代批次','batch_index',batch_index,'ssim:',skimage.metrics.structural_similarity(clear,noise_img2clear_img))
    print('测试集第',epoch,'迭代批次','batch_index',batch_index,'psnr:',skimage.metrics.peak_signal_noise_ratio(clear,noise_img2clear_img))
    img1 = clear.flatten()
    img2 = noise_img2clear_img.flatten()
    print('测试集第', epoch, '迭代', 'batch_index', batch_index, '的相关系数是:',np.corrcoef(img1,img2)[0,1])
    clear = Image.fromarray(clear)
    noise = Image.fromarray(noise)

    noise_img2clear_img = Image.fromarray(noise_img2clear_img)
    # noise_img2clear_img1 = Image.fromarray(noise_img2clear_img1)
    path = path + "/testepoch%d" % epoch
    if not os.path.exists(path):
        os.makedirs(path)
    clear.save(path + "/cleartest{}_{}.png".format(batch_epoch,batch_index))
    noise.save(path + "/noisetest{}_{}.png".format(batch_epoch,batch_index))
    noise_img2clear_img.save(path + "/noise_img2clear_imgtest{}_{}.png".format(batch_epoch,batch_index))


def sample_imagestest_jujiao(epoch, batch_epoch,batch_index, clear,noise, noise_img2clear_img, path):
    clear = clear.cpu().numpy()
    # noise_img2clear_img = noise_img2clear_img.cpu().numpy()
    noise_img2clear_img = noise_img2clear_img.detach().cpu().numpy()
    # noise_img2clear_img1 =noise_img2clear_img
    noise =noise.cpu().numpy()
    # for i in range(noise_img2clear_img1.shape[0]):
    #     for j in range(64):
    #         for k in range(64):
    #             if noise_img2clear_img1[i,:,j,k]>0.9:
    #                 noise_img2clear_img1[i,:,j,k]=noise_img2clear_img1[i,:,j,k]
    #             elif noise_img2clear_img1[i,:,j,k]>0.7:
    #                 noise_img2clear_img1[i, :, j, k] = 0.9
    #             elif noise_img2clear_img1[i,:,j,k]>0.3:
    #                 noise_img2clear_img1[i, :, j, k] = 0.1

    clear = clear.squeeze(0)
    noise_img2clear_img = noise_img2clear_img.squeeze(0)
    # noise_img2clear_img1 = noise_img2clear_img1.squeeze(0)
    noise = noise.squeeze(0)
    clear = np.uint8(255.0 * clear)
    noise = np.uint8(255.0 * noise)

    noise_img2clear_img = np.uint8(255.0 * noise_img2clear_img)
    # noise_img2clear_img1 = np.uint8(255.0 * noise_img2clear_img1)

    print('测试集第',epoch,'迭代批次','batch_index',batch_index,'ssim:',skimage.metrics.structural_similarity(clear,noise_img2clear_img))
    print('测试集第',epoch,'迭代批次','batch_index',batch_index,'psnr:',skimage.metrics.peak_signal_noise_ratio(clear,noise_img2clear_img))
    clear = Image.fromarray(clear)
    noise = Image.fromarray(noise)

    noise_img2clear_img = Image.fromarray(noise_img2clear_img)
    # noise_img2clear_img1 = Image.fromarray(noise_img2clear_img1)
    path = path + "/testepoch%d" % epoch
    if not os.path.exists(path):
        os.makedirs(path)
    clear.save(path + "/cleartest{}_{}.png".format(batch_epoch,batch_index))
    noise.save(path + "/noisetest{}_{}.png".format(batch_epoch,batch_index))
    noise_img2clear_img.save(path + "/noise_img2clear_imgtest{}_{}.png".format(batch_epoch,batch_index))
    # noise_img2clear_img1.save(path + "/noise_img2clear_imgtest_plus%d.png" % batch_index)



def sample_imagestestminist6464(epoch, batch_epoch,batch_index, clear,noise, noise_img2clear_img, path):
    clear = clear.cpu().numpy()
    # noise_img2clear_img = noise_img2clear_img.cpu().numpy()
    noise_img2clear_img = noise_img2clear_img.detach().cpu().numpy()
    # noise_img2clear_img1 =noise_img2clear_img
    noise =noise.cpu().numpy()
    # for i in range(noise_img2clear_img1.shape[0]):
    #     for j in range(64):
    #         for k in range(64):
    #             if noise_img2clear_img1[i,:,j,k]>0.9:
    #                 noise_img2clear_img1[i,:,j,k]=noise_img2clear_img1[i,:,j,k]
    #             elif noise_img2clear_img1[i,:,j,k]>0.7:
    #                 noise_img2clear_img1[i, :, j, k] = 0.9
    #             elif noise_img2clear_img1[i,:,j,k]>0.3:
    #                 noise_img2clear_img1[i, :, j, k] = 0.1

    clear = clear.squeeze(0)
    noise_img2clear_img = noise_img2clear_img.squeeze(0)
    # noise_img2clear_img1 = noise_img2clear_img1.squeeze(0)
    noise = noise.squeeze(0)
    clear = np.uint8(255.0 * clear)
    noise = np.uint8(255.0 * noise)

    noise_img2clear_img = np.uint8(255.0 * noise_img2clear_img)
    # noise_img2clear_img1 = np.uint8(255.0 * noise_img2clear_img1)

    print('测试集第',epoch,'迭代批次','batch_index',batch_index,'ssim:',skimage.metrics.structural_similarity(clear,noise_img2clear_img))
    print('测试集第',epoch,'迭代批次','batch_index',batch_index,'psnr:',skimage.metrics.peak_signal_noise_ratio(clear,noise_img2clear_img))
    clear = Image.fromarray(clear)
    noise = Image.fromarray(noise)

    noise_img2clear_img = Image.fromarray(noise_img2clear_img)
    # noise_img2clear_img1 = Image.fromarray(noise_img2clear_img1)
    path_input = path + "/testnoise%d" % epoch
    path_output = path + "/testclear%d" % epoch
    if not os.path.exists(path_input):
        os.makedirs(path_input)
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    # clear.save(path + "/cleartest{}_{}.png".format(batch_epoch,batch_index))

    noise.save(path_input + "/{}.png".format(batch_epoch))
    noise_img2clear_img.save(path_output + "/{}.png".format(batch_epoch))
    # noise_img2clear_img1.save(path + "/noise_img2clear_imgtest_plus%d.png" % batch_index)

def sample_imagestestminist6464_plus(epoch, batch_epoch,batch_index, clear,noise, noise_img2clear_img, path):
    clear = clear.cpu().numpy()
    # noise_img2clear_img = noise_img2clear_img.cpu().numpy()
    noise_img2clear_img = noise_img2clear_img.detach().cpu().numpy()
    # noise_img2clear_img1 =noise_img2clear_img
    noise =noise.detach().cpu().numpy()
    # for i in range(noise_img2clear_img1.shape[0]):
    #     for j in range(64):
    #         for k in range(64):
    #             if noise_img2clear_img1[i,:,j,k]>0.9:
    #                 noise_img2clear_img1[i,:,j,k]=noise_img2clear_img1[i,:,j,k]
    #             elif noise_img2clear_img1[i,:,j,k]>0.7:
    #                 noise_img2clear_img1[i, :, j, k] = 0.9
    #             elif noise_img2clear_img1[i,:,j,k]>0.3:
    #                 noise_img2clear_img1[i, :, j, k] = 0.1

    clear = clear.squeeze(0)
    noise_img2clear_img = noise_img2clear_img.squeeze(0)
    # noise_img2clear_img1 = noise_img2clear_img1.squeeze(0)
    noise = noise.squeeze(0)
    # noise_img2clear_img = noise_img2clear_img
    # noise_img2clear_img = (noise_img2clear_img -np.min(noise_img2clear_img))/ (np.max(noise_img2clear_img)-np.min(noise_img2clear_img))
    clear = np.uint8(255.0 * clear)
    noise = np.uint8(255.0 * noise)

    noise_img2clear_img = np.uint8(255.0 * noise_img2clear_img)
    # noise_img2clear_img1 = np.uint8(255.0 * noise_img2clear_img1)

    print('测试集第',epoch,'迭代批次','batch_index',batch_index,'ssim:',skimage.metrics.structural_similarity(clear,noise_img2clear_img))
    print('测试集第',epoch,'迭代批次','batch_index',batch_index,'psnr:',skimage.metrics.peak_signal_noise_ratio(clear,noise_img2clear_img))
    # clear = Image.fromarray(clear)
    noise = Image.fromarray(noise)

    noise_img2clear_img = Image.fromarray(noise_img2clear_img)
    # noise_img2clear_img1 = Image.fromarray(noise_img2clear_img1)
    # path_input = path + "/testnoise%d" % epoch
    path_output = path
    # if not os.path.exists(path_input):
    #     os.makedirs(path_input)
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    # clear.save(path + "/cleartest{}_{}.png".format(batch_epoch,batch_index))

    noise.save(path_output + "/a{}.png".format(batch_epoch))
    noise_img2clear_img.save(path_output + "/{}.png".format(batch_epoch))
    # noise_img2clear_img1.save(path + "/noise_img2clear_imgtest_plus%d.png" % batch_index)

def datasavetrain(model, loader, device,path,epoch):
# 存储数据
    for i, batch_data in enumerate(loader):
        noise = batch_data[0]
        clear = batch_data[1]
        # 处理第n个batch数据
        # x = 128  # 128+-64
        # y = 128 # 64+-32
        # noise = noise[:, :, x - 120:x + 120, y - 120:y + 120]
        # input1 = noise[:, :, 0:128, 0:128]
        # input2 = noise[:, :, 128:256, 0:128]
        # input3 = noise[:, :, 0:128, 128:256]
        # input4 = noise[:, :, 128:256, 128:256]
        # noise = torch.cat((input1, input2, input3, input4), dim=1)
        noise, clear = noise.to(device), clear.to(device)
        noise = noise.type(torch.cuda.FloatTensor)
        noise_img2clear_img = model(noise)
        for k in range(len(noise)):
             sample_imagestrain(epoch=epoch, batch_index=k + 1, clear=clear[k],
                                            noise_img2clear_img=noise_img2clear_img[k], path=path)
        break



def datasavetrain_TM(model, loader, device,path,epoch):
# 存储数据
    for i, batch_data in enumerate(loader):
        noise = batch_data[0]
        clear = batch_data[1]
        # 处理第n个batch数据
        # x = 128  # 128+-64
        # y = 128 # 64+-32
        # noise = noise[:, :, x - 120:x + 120, y - 120:y + 120]
        # input1 = noise[:, :, 0:128, 0:128]
        # input2 = noise[:, :, 128:256, 0:128]
        # input3 = noise[:, :, 0:128, 128:256]
        # input4 = noise[:, :, 128:256, 128:256]
        # noise = torch.cat((input1, input2, input3, input4), dim=1)
        noise, clear = noise.to(device), clear.to(device)
        clear = clear.type(torch.cuda.FloatTensor)
        noise_img2clear_img = model(clear)
        for k in range(len(noise)):
             sample_imagestrain(epoch=epoch, batch_index=k + 1, clear=noise[k],
                                            noise_img2clear_img=noise_img2clear_img[k], path=path)
        break


def datasaveval(model, loader, device,path,epoch):
# 存储数据
    for i, batch_data in enumerate(loader):
        noise = batch_data[0]
        clear = batch_data[1]
        # 处理第n个batch数据
        # x = 128  # 128+-64
        # y = 128 # 64+-32
        # noise = noise[:, :, x - 120:x + 120, y - 120:y + 120]
        # input1 = noise[:, :, 0:128, 0:128]
        # input2 = noise[:, :, 128:256, 0:128]
        # input3 = noise[:, :, 0:128, 128:256]
        # input4 = noise[:, :, 128:256, 128:256]
        # noise = torch.cat((input1, input2, input3, input4), dim=1)
        noise, clear = noise.to(device), clear.to(device)
        noise = noise.type(torch.cuda.FloatTensor)
        noise_img2clear_img = model(noise)
        for k in range(len(noise)):
             sample_imagesval(epoch=epoch, batch_index=k + 2, clear=clear[k],
                                            noise_img2clear_img=noise_img2clear_img[k], path=path)
        break
def datasavetest(model, loader, device,path,epoch):
# 存储数据
    criterion = torch.nn.MSELoss()
    for i, batch_data in enumerate(loader):
        noise = batch_data[0]
        clear = batch_data[1]
        # 处理第n个batch数据
        # x = 128  # 128+-64
        # y = 128 # 64+-32
        # noise = noise[:, :, x - 120:x + 120, y - 120:y + 120]
        # input1 = noise[:, :, 0:128, 0:128]
        # input2 = noise[:, :, 128:256, 0:128]
        # input3 = noise[:, :, 0:128, 128:256]
        # input4 = noise[:, :, 128:256, 128:256]
        # noise = torch.cat((input1, input2, input3, input4), dim=1)
        noise, clear = noise.to(device), clear.to(device)
        noise = noise.type(torch.cuda.FloatTensor)
        # noise = (noise - torch.min(noise)) \
        #                       / (torch.max(noise) - torch.min(noise))
        noise_img2clear_img = model(noise)
        # noise_img2clear_img = (noise_img2clear_img - torch.min(noise_img2clear_img)) \
        #                       / (torch.max(noise_img2clear_img) - torch.min(noise_img2clear_img))
        # for i in range(noise_img2clear_img.shape[0]):
        #     for j in range(40):
        #         for k in range(40):
        #             if noise_img2clear_img[i,:,j,k]>0.8:
        #                 noise_img2clear_img[i,:,j,k]=1
        #             else:
        #                 noise_img2clear_img[i, :, j, k] = 0
        # for p in range(len(noise)):
        loss = criterion(noise_img2clear_img, clear)
        print("迭代次数：", i+1, "损失值：", loss)
        for k in range(len(noise)):
             sample_imagestest(epoch=epoch , batch_epoch=i, batch_index=k + 1, clear=clear[k],noise =noise[k],
                                            noise_img2clear_img=noise_img2clear_img[k], path=path)
        # break




def datasavetest_jujiao_weitiao(model_netB,model_netA,loader, device,device_plus,path,epoch):
# 存储数据
    total_run_time =  0
    criterion = torch.nn.MSELoss()
    for i, batch_data in enumerate(loader):
        noise = batch_data[0]
        clear = batch_data[1]
        # 处理第n个batch数据
        # x = 128  # 128+-64
        # y = 128 # 64+-32
        # noise = noise[:, :, x - 120:x + 120, y - 120:y + 120]
        # input1 = noise[:, :, 0:128, 0:128]
        # input2 = noise[:, :, 128:256, 0:128]
        # input3 = noise[:, :, 0:128, 128:256]
        # input4 = noise[:, :, 128:256, 128:256]
        # noise = torch.cat((input1, input2, input3, input4), dim=1)
        noise, clear = noise.to(device), clear.to(device)
        noise = noise.type(torch.cuda.FloatTensor)
        start_time = time.time()
        noise_img2clear_img1 = model_netA(noise)
        end_time =time.time()
        run_time = end_time - start_time
        print("程序运行时间：%.6f秒"%run_time)
        total_run_time += run_time
        noise_img2clear_img1 = (noise_img2clear_img1 - torch.min(noise_img2clear_img1)) \
                              / (torch.max(noise_img2clear_img1) - torch.min(noise_img2clear_img1))
        noise_img2clear_img =  model_netB(noise_img2clear_img1)
        noise_img2clear_img= noise_img2clear_img.to(device)
        noise_img2clear_img = (noise_img2clear_img - torch.min(noise_img2clear_img)) \
                              / (torch.max(noise_img2clear_img) - torch.min(noise_img2clear_img))
        loss = criterion(noise_img2clear_img, noise)
        print("迭代次数：", i+1, "损失值：", loss)
        for k in range(len(noise)):
             sample_imagestest(epoch=epoch , batch_epoch=i, batch_index=k + 1, clear=noise[k],noise =noise_img2clear_img1[k],
                                            noise_img2clear_img=noise_img2clear_img[k], path=path)
        # break
    print("程序总共运行时间：%.6f秒" % total_run_time)


def datasavejujiao_test(model, loader, device,path,epoch):
# 存储数据
    criterion = torch.nn.MSELoss()
    for i, batch_data in enumerate(loader):
        noise = batch_data[0]
        clear = batch_data[1]
        # 处理第n个batch数据
        # x = 128  # 128+-64
        # y = 128 # 64+-32
        # noise = noise[:, :, x - 120:x + 120, y - 120:y + 120]
        # input1 = noise[:, :, 0:128, 0:128]
        # input2 = noise[:, :, 128:256, 0:128]
        # input3 = noise[:, :, 0:128, 128:256]
        # input4 = noise[:, :, 128:256, 128:256]
        # noise = torch.cat((input1, input2, input3, input4), dim=1)
        noise, clear = noise.to(device), clear.to(device)
        noise = noise.type(torch.cuda.FloatTensor)

        noise_img2clear_img = model(noise)

        # for i in range(noise_img2clear_img.shape[0]):
        #     for j in range(40):
        #         for k in range(40):
        #             if noise_img2clear_img[i,:,j,k]>0.8:
        #                 noise_img2clear_img[i,:,j,k]=1
        #             else:
        #                 noise_img2clear_img[i, :, j, k] = 0
        # for p in range(len(noise)):
        loss = criterion(noise_img2clear_img, clear)
        print("迭代次数：", i+1, "损失值：", loss)
        for k in range(len(noise)):
             sample_imagestest(epoch=epoch,batch_epoch=i, batch_index=k + 1, clear=clear[k],noise =noise[k],
                                            noise_img2clear_img=noise_img2clear_img[k], path=path)
        # break


def datasavetestminist6464(model, loader, device, path, epoch):
    # 存储数据
    p=0
    criterion = torch.nn.MSELoss()
    for i, batch_data in enumerate(loader):
        noise = batch_data[0]
        clear = batch_data[1]
        # 处理第n个batch数据
        # x = 128  # 128+-64
        # y = 128 # 64+-32
        # noise = noise[:, :, x - 120:x + 120, y - 120:y + 120]
        # input1 = noise[:, :, 0:128, 0:128]
        # input2 = noise[:, :, 128:256, 0:128]
        # input3 = noise[:, :, 0:128, 128:256]
        # input4 = noise[:, :, 128:256, 128:256]
        # noise = torch.cat((input1, input2, input3, input4), dim=1)
        noise, clear = noise.to(device), clear.to(device)
        noise = noise.type(torch.cuda.FloatTensor)
        noise_img2clear_img = model(noise)
        # for i in range(noise_img2clear_img.shape[0]):
        #     for j in range(40):
        #         for k in range(40):
        #             if noise_img2clear_img[i,:,j,k]>0.8:
        #                 noise_img2clear_img[i,:,j,k]=1
        #             else:
        #                 noise_img2clear_img[i, :, j, k] = 0
        # for p in range(len(noise)):
        loss = criterion(noise_img2clear_img, clear)
        print("迭代次数：", i + 1, "损失值：", loss)
        for k in range(len(noise)):
            p+=1
            sample_imagestestminist6464(epoch=epoch, batch_epoch=p, batch_index=k + 1, clear=clear[k], noise=noise[k],
                              noise_img2clear_img=noise_img2clear_img[k], path=path)
        # break
total_run_time =0
def datasavetestminist6464_plus(model, loader, device, path, epoch):
    # 存储数据
    p=0
    total_run_time =0
    criterion = torch.nn.MSELoss()
    for i, batch_data in enumerate(loader):
        noise = batch_data[0]
        clear = batch_data[1]
        # 处理第n个batch数据
        # x = 128  # 128+-64
        # y = 128 # 64+-32
        # noise = noise[:, :, x - 120:x + 120, y - 120:y + 120]
        # input1 = noise[:, :, 0:128, 0:128]
        # input2 = noise[:, :, 128:256, 0:128]
        # input3 = noise[:, :, 0:128, 128:256]
        # input4 = noise[:, :, 128:256, 128:256]
        # noise = torch.cat((input1, input2, input3, input4), dim=1)
        # start_time =time.time()
        noise, clear = noise.to(device), clear.to(device)
        noise = noise.type(torch.cuda.FloatTensor)
        noise = (noise-noise.min())/ (noise.max()-noise.min())
        noise_img2clear_img = model(noise)
        # noise_img2clear_img = (noise_img2clear_img-noise_img2clear_img.min())/\
        #                       (noise_img2clear_img.max()-noise_img2clear_img.min())
        # for i in range(noise_img2clear_img.shape[0]):
        #     for j in range(40):
        #         for k in range(40):
        #             if noise_img2clear_img[i,:,j,k]>0.8:
        #                 noise_img2clear_img[i,:,j,k]=1
        #             else:
        #                 noise_img2clear_img[i, :, j, k] = 0
        # for p in range(len(noise)):
        loss = criterion(noise_img2clear_img, clear)
        print("迭代次数：", i + 1, "损失值：", loss)
        # end_time =time.time()
        # run_time = end_time - start_time
        # print("程序运行时间：%.6f秒"%run_time)
        # total_run_time += run_time
        for k in range(len(noise)):
            p+=1
            sample_imagestestminist6464_plus(epoch=epoch, batch_epoch=p, batch_index=k + 1, clear=clear[k], noise=noise[k],
                              noise_img2clear_img=noise_img2clear_img[k], path=path)
        # break
    # print("程序总共运行时间：%.6f秒" % total_run_time)

      # break

def datasavetestminist6464_focus128(model, loader, device, path, epoch,tm,):
    # 存储数据
    p=0
    path = path + "/epoch%d" % epoch
    for i, batch_data in enumerate(loader):
        noise = batch_data[0]
        clear = batch_data[1]
        noise, clear = noise.to(device), clear.to(device)
        tm=tm.to(device)
        noise = noise.type(torch.cuda.FloatTensor)
        noise_img2clear_img = model(noise)


        for k in range(len(noise)):
                p+=1
                output = (noise_img2clear_img[k] - torch.min(noise_img2clear_img[k])) / (
                            torch.max(noise_img2clear_img[k]) - torch.min(noise_img2clear_img[k]))
                # 确保在计算过程中保持 requires_grad=True
                output = 255.0 * output  # 归一化到 [0, 255]

                # 保持为 FloatTensor，而不是转换为 uint8，这样梯度可以继续计算
                output = output.to(torch.float32)

                # 将 output 转换为合适的形状进行矩阵运算
                data = output.reshape(1, -1).T  # 按行展开并转置
                # 将 data 转换为复数张量
                data_complex = torch.view_as_complex(torch.stack((data, torch.zeros_like(data)), dim=-1))
                # 在 PyTorch 中进行矩阵运算，而不是转换为 NumPy
                result = torch.matmul(tm, data_complex)  # 使用 PyTorch 的 matmul
                result = result.T

                result = result.reshape(1, 128, 128)
                result1 = (torch.abs(result) ** 2) / torch.max(torch.abs(result) ** 2)  # 归一化
                sample_imagestestminist6464_plus(epoch=epoch, batch_epoch=p, batch_index=k + 1, clear=clear[k], noise=noise_img2clear_img[k],
                                  noise_img2clear_img=result1, path=path)
            # break
        # print("程序总共运行时间：%.6f秒" % total_run_time)
