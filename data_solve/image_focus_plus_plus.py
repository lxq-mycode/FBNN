# import numpy as np
# import matplotlib.pyplot as plt
#
#
# # Function to create and save an image with a square at a specified position
# def create_image_with_square(center_x, center_y, filename):
#     # Create a 256x256 black image
#     image = np.zeros((256, 256), dtype=np.uint8)
#
#     # Define the size of the square
#     square_size = 8
#
#     # Calculate the start and end points of the square
#     start_x = center_x - square_size // 2
#     end_x = start_x + square_size
#     start_y = center_y - square_size // 2
#     end_y = start_y + square_size
#
#     # Set the pixel values of the square to 255
#     image[start_y:end_y, start_x:end_x] = 255
#
#     # Save the image
#     plt.imsave(filename, image, cmap='gray')
#
#
# # Define center coordinates and corresponding filenames
# centers = [(128, 128), (192, 128), (128, 192), (128, 64), (64, 128)]
# filenames = ["/home/jnu/下载/1.png", "/home/jnu/下载/2.png", "/home/jnu/下载/3.png", "/home/jnu/下载/4.png", "/home/jnu/下载/5.png"]
#
# # Create and save images
# for (center_x, center_y), filename in zip(centers, filenames):
#     create_image_with_square(center_x, center_y, filename)




import numpy as np
import matplotlib.pyplot as plt
# from PIL import Image, ImageDraw, ImageFont
# Function to create and save an image with a square at a specified position and size
def create_image_with_square(center_x, center_y, square_size, filename):
    # Create a 256x256 black image
    image = np.zeros((256, 256), dtype=np.uint8)

    # Calculate the start and end points of the square
    start_x = center_x - square_size // 2
    end_x = start_x + square_size
    start_y = center_y - square_size // 2
    end_y = start_y + square_size

    # Set the pixel values of the square to 255
    image[start_y:end_y, start_x:end_x] = 255

    # Save the image
    plt.imsave(filename, image, cmap='gray')

# Define the center coordinates for all images (centered square)
center = (128, 128)
# Define the sizes of the squares
square_sizes = [8, 10, 12,14,16]
filenames = ["/media/jnu/data2/10_25shiyanzhengli/jujiao/dot/LQ/1.png",
             "/media/jnu/data2/10_25shiyanzhengli/jujiao/dot/LQ/2.png",
             "/media/jnu/data2/10_25shiyanzhengli/jujiao/dot/LQ/3.png",
             "/media/jnu/data2/10_25shiyanzhengli/jujiao/dot/LQ/4.png",
             "/media/jnu/data2/10_25shiyanzhengli/jujiao/dot/LQ/5.png"]

# Create and save images
for square_size, filename in zip(square_sizes, filenames):
    create_image_with_square(center[0], center[1], square_size, filename)