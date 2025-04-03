import numpy as np
from PIL import Image
import os

# Function to create a white image
def create_white_image(size):
    return np.ones(size, dtype=np.uint8) * 255

# Define the new image size and output directory
image_size = (64, 64)  # 64x64 pixels
output_dir = '/media/jnu/SUCCESS/data_6_28/white_test/HQ'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Generate and save 5 white images
for i in range(5):
    image = create_white_image(image_size)
    img = Image.fromarray(image, mode='L')
    output_path = os.path.join(output_dir, f'white_image_{i+1}.png')
    img.save(output_path)

print(f"Images saved to {output_dir}")