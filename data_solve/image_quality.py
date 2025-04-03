import skimage.metrics
import cv2
image_path1 = "/media/jnu/data2/model_new/conv_Linear_net3/testepoch160/noise_img2clear_imgtest3.png"
image_path2 = "/media/jnu/data2/model_new/conv_Linear_net3/testepoch160/cleartest3.png"

clear_data = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
noise_data = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)

print('ssim:',skimage.metrics.structural_similarity(clear_data, noise_data))
print('psnr:',skimage.metrics.peak_signal_noise_ratio(noise_data, clear_data))
