import cv2
import numpy as np
from PIL import Image


# test_clear_path = '/media/jnu/SUCCESS/data_6_28/jujiao/dian/net/LQ'
# test_noise = '/media/jnu/SUCCESS/data_6_28/jujiao/dian/net/yuce/testmodel_clear1'

net_path = '/media/jnu/SUCCESS/data_7_5/TM/test/net/LQ/testmodel_clear1/5.png'
prvamp_path =  '/media/jnu/SUCCESS/data_7_5/TM/test/prvamp/LQ/5.png'
origial_path =  '/media/jnu/SUCCESS/data_7_5/TM/test/true/5.png'


net = Image.open(net_path)
prvamp = Image.open(prvamp_path)
origial = Image.open(origial_path)


net_np = np.array(net).flatten()
prvamp_np = np.array(prvamp).flatten()
origial_np = np.array(origial).flatten()

net_mean = np.mean(net_np)
prvamp_mean = np.mean(prvamp_np)
origial_mean = np.mean(origial_np)

net_var= np.var(net_np)
prvamp_var = np.var(prvamp_np)
origial_var= np.var(origial_np)



corr_coef_net_o = np.corrcoef(net_np,origial_np)[0,1]
corr_coef_prvamp_o = np.corrcoef(prvamp_np,origial_np)[0,1]
print("net和原始图片的相关系数是",corr_coef_net_o)
print("prvamp和原始图片的相关系数是",corr_coef_prvamp_o)
print("net的均值和方差是",net_mean,net_var)
print("prvamp的均值和方差是",prvamp_mean,prvamp_var)
print("原始图片的均值和方差是",origial_mean,origial_var)

# for idx in range(1, 6):
#     c_path = ('%s/%d.png' % (test_clear_path, idx))
#     n_path = ('%s/%d.png' % (test_noise, idx))
#     image1 = Image.open(c_path)
#     image2 = Image.open(n_path)
#     img1 = np.array(image1).flatten()
#     img2 = np.array(image2).flatten()
#
#
#     corr_coef = np.corrcoef(img1,img2)[0,1]
#     print("第",idx,"张图片的相关系数是",corr_coef)
#
'''
net和原始图片的相关系数是 0.7701262242278455
prvamp和原始图片的相关系数是 0.7236151258435741
net的均值和方差是 15.325668334960938 508.6200906482991
prvamp的均值和方差是 57.6337890625 1062.6629781723022
原始图片的均值和方差是 41.59031677246094 1570.6940523532685


net和原始图片的相关系数是 0.87369788385644
prvamp和原始图片的相关系数是 0.8427521637425602
net的均值和方差是 28.956390380859375 1567.2145899003372
prvamp的均值和方差是 63.92893981933594 1450.753022350138
原始图片的均值和方差是 39.31304931640625 1537.8677899204195

net和原始图片的相关系数是 0.7892988972795397
prvamp和原始图片的相关系数是 0.7596576935283408
net的均值和方差是 6.7245941162109375 166.48877257085405
prvamp的均值和方差是 66.61393737792969 1201.4149357543793
原始图片的均值和方差是 19.97149658203125 395.6865349672735

net和原始图片的相关系数是 0.8344519864548203
prvamp和原始图片的相关系数是 0.8100147127914855
net的均值和方差是 13.293746948242188 572.0143749618437
prvamp的均值和方差是 57.818359375 1118.780605316162
原始图片的均值和方差是 32.519195556640625 979.417783386074


net和原始图片的相关系数是 0.7941400991922685
prvamp和原始图片的相关系数是 0.7558367764628705
net的均值和方差是 26.743850708007812  1081.3874057286885
prvamp的均值和方差是 63.8773193359375 1377.578180655837
原始图片的均值和方差是 37.08293151855469 1223.8958170653787

net和原始图片的相关系数是 0.8233464973660843
prvamp和原始图片的相关系数是 0.8260561411177365
net的均值和方差是 69.89064025878906 2465.871081828838
prvamp的均值和方差是 66.34286499023438 1499.1246005808935
原始图片的均值和方差是 50.015777587890625 1923.7924451595172

'''


