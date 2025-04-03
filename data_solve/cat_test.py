import torch
import matplotlib.pyplot as plt

x1_output = torch.randn(32,32)
x2_output = torch.randn(32,32)
x3_output = torch.randn(32,32)
x4_output = torch.randn(32,32)
ouput_top_row = torch.cat((x1_output, x2_output), dim=1)
ouput_bottom_row = torch.cat((x3_output, x4_output), dim=1)
output = torch.cat((ouput_top_row, ouput_bottom_row), dim=0)

output = output.numpy()
plt.imshow(output,cmap='gray')
plt.axis('off')
plt.show()