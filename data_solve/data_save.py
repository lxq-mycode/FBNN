import numpy as np
from PIL import Image
import os
import torch
from data_solve import optimizer
import skimage.metrics
import cv2


def sample_imagestrain_willer_plus_fuliye(epoch ,batch_index, clear, noise_img2clear_img, path):
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

def sample_imagestrain_willer_plus(epoch ,batch_index, clear, noise_img2clear_img, path):
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



def sample_imagesval_willer_plus(epoch, batch_index, clear, noise_img2clear_img, path):
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



def sample_imagesval_willer_plus_fuliye(epoch, batch_index, clear, noise_img2clear_img, path):
        clear = clear.cpu().numpy()
        # noise_img2clear_img = noise_img2clear_img.cpu().numpy()
        noise_img2clear_img = noise_img2clear_img.detach().cpu().numpy()

        # writer.add_image("clearval", clear, epoch)
        # writer.add_image("noiseval", noise, epoch)
        # writer.add_image("noise_img2clear_imgval", noise_img2clear_img, epoch)
        clear = clear.squeeze(0)
        noise_img2clear_img = noise_img2clear_img.squeeze(0)
        clear = np.uint8(clear * 255)
        noise_img2clear_img = np.uint8(noise_img2clear_img * 255)
        print('验证集第', epoch, '迭代', 'batch_index', batch_index, 'ssim:',
              skimage.metrics.structural_similarity(clear, noise_img2clear_img))
        print('验证集第', epoch, '迭代', 'batch_index', batch_index, 'psnr:',
              skimage.metrics.peak_signal_noise_ratio(clear, noise_img2clear_img))
        img1 = clear.flatten()
        img2 = noise_img2clear_img.flatten()
        print('验证集第', epoch, '迭代', 'batch_index', batch_index, '的相关系数是:', np.corrcoef(img1, img2)[0, 1])
        clear = Image.fromarray(clear)
        noise_img2clear_img = Image.fromarray(noise_img2clear_img)

        path = path + "/valepoch%d" % epoch
        if not os.path.exists(path):
            os.makedirs(path)
        clear.save(path + "/clearval%d.png" % batch_index)
        noise_img2clear_img.save(path + "/noise_img2clear_imgval%d.png" % batch_index)



def sample_imagestrain_willer(epoch ,batch_index, clear, noise, noise_img2clear_img, path):
    clear = clear.cpu().numpy()
    noise = noise.cpu().numpy()
    # noise_img2clear_img = noise_img2clear_img.cpu().numpy()
    noise_img2clear_img = noise_img2clear_img.detach().cpu().numpy()

    # writer.add_image("cleartrain", clear, epoch)
    # writer.add_image("noisetain", noise, epoch)
    # writer.add_image("noise_img2clear_imgtrain", noise_img2clear_img, epoch)
    clear = np.uint8(255.0 * clear)
    noise = np.uint8(noise*255)
    noise_img2clear_img = np.uint8(255.0 * noise_img2clear_img)

    clear = np.transpose(clear, (1, 2, 0))
    noise = np.transpose(noise, (1, 2, 0))
    noise_img2clear_img = np.transpose(noise_img2clear_img, (1, 2, 0))

    print('训练集第',epoch,'迭代','batch_index',batch_index,'ssim:',optimizer.ssim(clear,noise_img2clear_img))
    print('训练集第',epoch,'迭代','batch_index',batch_index,'psnr:',optimizer.calculate_psnr1(clear,noise_img2clear_img))
    clear = Image.fromarray(clear)
    noise = Image.fromarray(noise)
    noise_img2clear_img = Image.fromarray(noise_img2clear_img)

    # 保存图片
    path =path + "/trainepoch%d" % epoch
    if not os.path.exists(path):
        os.makedirs(path)
    clear.save(path + "/cleartrain%d.png" % batch_index)
    noise.save(path + "/noisetrain%d.png" % batch_index)
    noise_img2clear_img.save(path + "/noise_img2clear_imgtrain%d.png" % batch_index)

def sample_imagestrain(epoch ,batch_index, clear, noise, noise_img2clear_img, path):
    clear = clear.cpu().numpy()
    noise = noise.cpu().numpy()
    # noise_img2clear_img = noise_img2clear_img.cpu().numpy()
    noise_img2clear_img = noise_img2clear_img.detach().cpu().numpy()

    clear = clear.squeeze(0)
    noise = noise.squeeze(0)
    noise_img2clear_img = noise_img2clear_img.squeeze(0)
    # writer.add_image("cleartrain", clear, epoch)
    # writer.add_image("noisetain", noise, epoch)
    # writer.add_image("noise_img2clear_imgtrain", noise_img2clear_img, epoch)
    clear = np.uint8(255.0 * clear)
    noise = np.uint8(noise*255)
    noise_img2clear_img = np.uint8(255.0 * noise_img2clear_img)
    print('训练集第',epoch,'迭代','batch_index',batch_index,'ssim:',optimizer.ssim(clear,noise_img2clear_img))
    print('训练集第',epoch,'迭代','batch_index',batch_index,'psnr:',optimizer.calculate_psnr1(clear,noise_img2clear_img))
    clear = Image.fromarray(clear)
    noise = Image.fromarray(noise)
    noise_img2clear_img = Image.fromarray(noise_img2clear_img)

    # 保存图片
    path =path + "/trainepoch%d" % epoch
    if not os.path.exists(path):
        os.makedirs(path)
    clear.save(path + "/cleartrain%d.png" % batch_index)
    noise.save(path + "/noisetrain%d.png" % batch_index)
    noise_img2clear_img.save(path + "/noise_img2clear_imgtrain%d.png" % batch_index)



def sample_imagesval(epoch, batch_index, clear, noise, noise_img2clear_img, path):
    clear = clear.cpu().numpy()
    noise = noise.cpu().numpy()
    # noise_img2clear_img = noise_img2clear_img.cpu().numpy()
    noise_img2clear_img = noise_img2clear_img.detach().cpu().numpy()

    # writer.add_image("clearval", clear, epoch)
    # writer.add_image("noiseval", noise, epoch)
    # writer.add_image("noise_img2clear_imgval", noise_img2clear_img, epoch)
    clear = clear.squeeze(0)
    noise = noise.squeeze(0)
    noise_img2clear_img = noise_img2clear_img.squeeze(0)
    clear = np.uint8(255.0 * clear)
    noise = np.uint8(noise*255)
    noise_img2clear_img = np.uint8(255.0 * noise_img2clear_img)
    print('验证集第',epoch,'迭代','batch_index',batch_index,'ssim:',optimizer.ssim(clear,noise_img2clear_img))
    print('验证集第',epoch,'迭代','batch_index',batch_index,'psnr:',optimizer.calculate_psnr1(clear,noise_img2clear_img))
    clear = Image.fromarray(clear)
    noise = Image.fromarray(noise)
    noise_img2clear_img = Image.fromarray(noise_img2clear_img)

    path = path + "/valepoch%d" % epoch
    if not os.path.exists(path):
        os.makedirs(path)
    clear.save(path + "/clearval%d.png" % batch_index)
    noise.save(path + "/noiseval%d.png" % batch_index)
    noise_img2clear_img.save(path + "/noise_img2clear_imgval%d.png" % batch_index)


def sample_imagestest(epoch, batch_index, clear, noise, noise_img2clear_img, path):
    clear = clear.cpu().numpy()
    noise = noise.cpu().numpy()
    # noise_img2clear_img = noise_img2clear_img.cpu().numpy()
    noise_img2clear_img = noise_img2clear_img.detach().cpu().numpy()
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
    noise = noise.squeeze(0)
    noise_img2clear_img = noise_img2clear_img.squeeze(0)

    clear = np.uint8(255.0 * clear)
    noise = np.uint8(noise*255)
    noise_img2clear_img = np.uint8(255.0 * noise_img2clear_img)

    print('测试集第',epoch,'迭代批次','batch_index',batch_index,'ssim:',optimizer.ssim(clear,noise_img2clear_img))
    print('测试集第',epoch,'迭代批次','batch_index',batch_index,'psnr:',optimizer.calculate_psnr1(clear,noise_img2clear_img))
    clear = Image.fromarray(clear)
    noise = Image.fromarray(noise)
    noise_img2clear_img = Image.fromarray(noise_img2clear_img)


    path = path + "/testepoch%d" % epoch
    if not os.path.exists(path):
        os.makedirs(path)
    clear.save(path + "/cleartest%d.png" % batch_index)
    noise.save(path + "/noisetest%d.png" % batch_index)
    noise_img2clear_img.save(path + "/noise_img2clear_imgtest%d.png" % batch_index)

def sample_imagestest_willer(epoch, batch_index, clear, noise, noise_img2clear_img, path):
    clear = clear.cpu().numpy()
    noise = noise.cpu().numpy()
    # noise_img2clear_img = noise_img2clear_img.cpu().numpy()
    noise_img2clear_img = noise_img2clear_img.detach().cpu().numpy()
    # for i in range(noise_img2clear_img1.shape[0]):
    #     for j in range(64):
    #         for k in range(64):
    #             if noise_img2clear_img1[i,:,j,k]>0.9:
    #                 noise_img2clear_img1[i,:,j,k]=noise_img2clear_img1[i,:,j,k]
    #             elif noise_img2clear_img1[i,:,j,k]>0.7:
    #                 noise_img2clear_img1[i, :, j, k] = 0.9
    #             elif noise_img2clear_img1[i,:,j,k]>0.3:
    #                 noise_img2clear_img1[i, :, j, k] = 0.1

    clear = np.uint8(255.0 * clear)
    noise = np.uint8(noise*255)
    noise_img2clear_img = np.uint8(255.0 * noise_img2clear_img)
    clear = np.transpose(clear, (1, 2, 0))
    noise = np.transpose(noise, (1, 2, 0))
    noise_img2clear_img = np.transpose(noise_img2clear_img, (1, 2, 0))

    print('测试集第',epoch,'迭代批次','batch_index',batch_index,'ssim:',optimizer.ssim(clear,noise_img2clear_img))
    print('训练集第',epoch,'迭代批次','batch_index',batch_index,'psnr:',optimizer.calculate_psnr1(clear,noise_img2clear_img))
    noise_img2clear_img = cv2.cvtColor(noise_img2clear_img, cv2.COLOR_RGB2BGR)  # 将通道顺序从RGB转换为BGR
    clear = cv2.cvtColor(clear, cv2.COLOR_RGB2BGR)  # 将通道顺序从RGB转换为BGR
    clear = Image.fromarray(clear)
    noise = Image.fromarray(noise)
    noise_img2clear_img = Image.fromarray(noise_img2clear_img)


    path = path + "/testepoch%d" % epoch
    if not os.path.exists(path):
        os.makedirs(path)
    clear.save(path + "/cleartest%d.png" % batch_index)
    noise.save(path + "/noisetest%d.png" % batch_index)
    noise_img2clear_img.save(path + "/noise_img2clear_imgtest%d.png" % batch_index)



def datasavetrain(model, loader, device,path,epoch):
# 存储数据
    for i, batch_data in enumerate(loader):
        noise = batch_data[0]
        clear = batch_data[1]
        # 处理第n个batch数据
        noise, clear = noise.to(device), clear.to(device)
        noise = noise.type(torch.cuda.FloatTensor)
        # noise = noise - 0.07339818
        noise_img2clear_img = model(noise)
        for k in range(len(noise)):
             sample_imagestrain(epoch=epoch, batch_index=k + 1, clear=clear[k], noise=noise[k],
                                            noise_img2clear_img=noise_img2clear_img[k], path=path)
        break

def datasavetrain_willer(model, loader, device,path,epoch):
# 存储数据
    for i, batch_data in enumerate(loader):
        noise = batch_data[0]
        clear = batch_data[1]
        # 处理第n个batch数据
        noise, clear = noise.to(device), clear.to(device)
        noise = noise.type(torch.cuda.FloatTensor)
        # noise = noise - 0.07339818
        noise_img2clear_img = model(noise)
        for k in range(len(noise)):
             sample_imagestrain_willer(epoch=epoch, batch_index=k + 1, clear=clear[k], noise=noise[k],
                                            noise_img2clear_img=noise_img2clear_img[k], path=path)
        break
def datasavetrain_willer_plus(model, loader, device,path,epoch):
# 存储数据
    for i, batch_data in enumerate(loader):
        noise = batch_data[0]
        clear = batch_data[1]
        # 处理第n个batch数据
        noise, clear = noise.to(device), clear.to(device)
        noise = noise.type(torch.cuda.FloatTensor)
        # noise = noise - 0.07339818
        noise_img2clear_img = model(noise)
        for k in range(len(noise)):
             sample_imagestrain_willer_plus(epoch=epoch, batch_index=k + 1, clear=clear[k],
                                            noise_img2clear_img=noise_img2clear_img[k], path=path)
        break

def datasavetrain_willer_plus_fuliye(model, loader, device,path,epoch):
# 存储数据
    for i, batch_data in enumerate(loader):
        noise = batch_data[0]
        clear = batch_data[1]
        # 处理第n个batch数据
        noise, clear = noise.to(device), clear.to(device)
        noise = noise.type(torch.cuda.FloatTensor)
        # noise = noise - 0.07339818
        noise_img2clear_img = model(noise)
        for k in range(len(noise)):
             sample_imagestrain_willer_plus_fuliye(epoch=epoch, batch_index=k + 1, clear=clear[k],
                                            noise_img2clear_img=noise_img2clear_img[k], path=path)
        break

def datasaveval(model, loader, device,path,epoch):
# 存储数据
    for i, batch_data in enumerate(loader):
        noise = batch_data[0]
        clear = batch_data[1]
        # 处理第n个batch数据
        noise, clear = noise.to(device), clear.to(device)
        noise = noise.type(torch.cuda.FloatTensor)
        # noise = noise - 0.07339818
        noise_img2clear_img = model(noise)
        noise_img2clear_img = (noise_img2clear_img-noise_img2clear_img.min())/\
                              (noise_img2clear_img.max()-noise_img2clear_img.min())
        for k in range(len(noise)):
             sample_imagesval(epoch=epoch, batch_index=k + 1, clear=clear[k], noise=noise[k],
                                            noise_img2clear_img=noise_img2clear_img[k], path=path)
        break
def datasavetest(model, loader, device,path,epoch):
# 存储数据
    criterion = torch.nn.MSELoss()
    for i, batch_data in enumerate(loader):
        noise = batch_data[0]
        clear = batch_data[1]
        # 处理第n个batch数据
        noise, clear = noise.to(device), clear.to(device)
        noise = noise.type(torch.cuda.FloatTensor)
        # noise = noise - 0.07339818
        noise_img2clear_img = model(noise)
        # for i in range(noise_img2clear_img.shape[0]):
        #     for j in range(40):
        #         for k in range(40):
        #             if noise_img2clear_img[i,:,j,k]>0.8:
        #                 noise_img2clear_img[i,:,j,k]=1
        #             else:
        #                 noise_img2clear_img[i, :, j, k] = 0

        loss = criterion(noise_img2clear_img, clear)
        print("迭代次数：", i+1, "损失值：", loss)
        for k in range(len(noise)):
             sample_imagestest(epoch=epoch, batch_index=k + 1, clear=clear[k], noise=noise[k],
                                            noise_img2clear_img=noise_img2clear_img[k], path=path)
        break

def datasavetest(model, loader, device,path,epoch):
# 存储数据
    criterion = torch.nn.MSELoss()
    for i, batch_data in enumerate(loader):
        noise = batch_data[0]
        clear = batch_data[1]
        # 处理第n个batch数据
        noise, clear = noise.to(device), clear.to(device)
        noise = noise.type(torch.cuda.FloatTensor)
        # noise = noise - 0.07339818
        noise_img2clear_img = model(noise)
        # for i in range(noise_img2clear_img.shape[0]):
        #     for j in range(40):
        #         for k in range(40):
        #             if noise_img2clear_img[i,:,j,k]>0.8:
        #                 noise_img2clear_img[i,:,j,k]=1
        #             else:
        #                 noise_img2clear_img[i, :, j, k] = 0

        loss = criterion(noise_img2clear_img, clear)
        print("迭代次数：", i+1, "损失值：", loss)
        for k in range(len(noise)):
             sample_imagestest(epoch=epoch, batch_index=k + 1, clear=clear[k], noise=noise[k],
                                            noise_img2clear_img=noise_img2clear_img[k], path=path)
        break

def datasavetest_willer(model, loader, device,path,epoch):
# 存储数据
    criterion = torch.nn.MSELoss()
    for i, batch_data in enumerate(loader):
        noise = batch_data[0]
        clear = batch_data[1]
        # 处理第n个batch数据
        noise, clear = noise.to(device), clear.to(device)
        noise = noise.type(torch.cuda.FloatTensor)
        # noise = noise - 0.07339818
        noise_img2clear_img = model(noise) # 3*400*192
        # for i in range(noise_img2clear_img.shape[0]):
        #     for j in range(40):
        #         for k in range(40):
        #             if noise_img2clear_img[i,:,j,k]>0.8:
        #                 noise_img2clear_img[i,:,j,k]=1
        #             else:
        #                 noise_img2clear_img[i, :, j, k] = 0

        loss = criterion(noise_img2clear_img, clear)
        print("迭代次数：", i+1, "损失值：", loss)
        for k in range(len(noise)):
             sample_imagestest_willer(epoch=epoch, batch_index=k + 1, clear=clear[k], noise=noise[k],
                                            noise_img2clear_img=noise_img2clear_img[k], path=path)
        break


def datasavetest_willer_plus(model, loader, device,path,epoch):
# 存储数据
    criterion = torch.nn.MSELoss()
    for i, batch_data in enumerate(loader):
        noise = batch_data[0]
        clear = batch_data[1]
        # 处理第n个batch数据
        noise, clear = noise.to(device), clear.to(device)
        noise = noise.type(torch.cuda.FloatTensor)
        # noise = noise - 0.07339818
        noise_img2clear_img = model(noise) # 3*400*192
        # for i in range(noise_img2clear_img.shape[0]):
        #     for j in range(40):
        #         for k in range(40):
        #             if noise_img2clear_img[i,:,j,k]>0.8:
        #                 noise_img2clear_img[i,:,j,k]=1
        #             else:
        #                 noise_img2clear_img[i, :, j, k] = 0

        loss = criterion(noise_img2clear_img, clear)
        print("迭代次数：", i+1, "损失值：", loss)
        for k in range(len(noise)):
             sample_imagesval_willer_plus(epoch=epoch, batch_index=k + 1, clear=clear[k],
                                            noise_img2clear_img=noise_img2clear_img[k], path=path)
        break


def datasavetest_willer_plus_fuliye(model, loader, device, path, epoch):
            # 存储数据
        criterion = torch.nn.MSELoss()
        for i, batch_data in enumerate(loader):
            noise = batch_data[0]
            clear = batch_data[1]
            # 处理第n个batch数据
            noise, clear = noise.to(device), clear.to(device)
            noise = noise.type(torch.cuda.FloatTensor)
            # noise = noise - 0.07339818
            noise_img2clear_img = model(noise)  # 3*400*192
            # for i in range(noise_img2clear_img.shape[0]):
            #     for j in range(40):
            #         for k in range(40):
            #             if noise_img2clear_img[i,:,j,k]>0.8:
            #                 noise_img2clear_img[i,:,j,k]=1
            #             else:
            #                 noise_img2clear_img[i, :, j, k] = 0

            loss = criterion(noise_img2clear_img, clear)
            print("迭代次数：", i + 1, "损失值：", loss)
            for k in range(len(noise)):
                sample_imagesval_willer_plus_fuliye(epoch=epoch, batch_index=k + 1, clear=clear[k],
                                             noise_img2clear_img=noise_img2clear_img[k], path=path)
            break
