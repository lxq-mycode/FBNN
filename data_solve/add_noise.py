import os
import torch
from PIL import Image
from torchvision import transforms


def add_noise_to_image(image, noise_factor=0.1):
    """给图像添加高斯噪声"""
    noise = torch.randn(image.size(1), image.size(2), device=image.device) * noise_factor
    noisy_image = image + noise
    noisy_image = torch.clamp(noisy_image, 0, 1)
    return noisy_image


def process_images(directory, output_directory, start, end, noise_factor=0.1):
    # 确保输出目录存在
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 图片转换为张量
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 归一化到[-1, 1]
    ])

    for i in range(start, end + 1):
        filename = f"{i}.png"
        img_path = os.path.join(directory, filename)

        if not os.path.exists(img_path):
            print(f"文件 {filename} 不存在，跳过。")
            continue

        # 读取图片
        image = Image.open(img_path).convert('L')  # 确保是灰度图

        # 将图片转换为张量
        image_tensor = transform(image).unsqueeze(0)  # 添加批次维度

        # 添加噪声
        noisy_image_tensor = add_noise_to_image(image_tensor, noise_factor).squeeze(0)

        # 将张量转换回图片
        noisy_image = transforms.ToPILImage()(noisy_image_tensor)

        # 保存带噪声的图片
        noisy_image.save(os.path.join(output_directory, filename))


# 使用示例
directory = '/media/jnu/data2/10_25shiyanzhengli/jujiao/dot/LQ'  # 图片目录路径
output_directory = '/media/jnu/data2/10_25shiyanzhengli/jujiao/dot/LQ_1'  # 带噪声图片保存目录路径
process_images(directory, output_directory, 1, 5, noise_factor=0.1)