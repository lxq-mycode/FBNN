from torch.optim.optimizer import Optimizer
import math
import cv2
import numpy as np
import torch

#
# class Lion(Optimizer):
#   r"""Implements Lion algorithm."""
#
#   def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
#     """Initialize the hyperparameters.
#
#     Args:
#       params (iterable): iterable of parameters to optimize or dicts defining
#         parameter groups
#       lr (float, optional): learning rate (default: 1e-4)
#       betas (Tuple[float, float], optional): coefficients used for computing
#         running averages of gradient and its square (default: (0.9, 0.99))
#       weight_decay (float, optional): weight decay coefficient (default: 0)
#     """
#
#     if not 0.0 <= lr:
#       raise ValueError('Invalid learning rate: {}'.format(lr))
#     if not 0.0 <= betas[0] < 1.0:
#       raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
#     if not 0.0 <= betas[1] < 1.0:
#       raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
#     defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
#     super().__init__(params, defaults)
#
#   @torch.no_grad()
#   def step(self, closure=None):
#     """Performs a single optimization step.
#
#     Args:
#       closure (callable, optional): A closure that reevaluates the model
#         and returns the loss.
#
#     Returns:
#       the loss.
#     """
#     loss = None
#     if closure is not None:
#       with torch.enable_grad():
#         loss = closure()
#
#     for group in self.param_groups:
#       for p in group['params']:
#         if p.grad is None:
#           continue
#
#         # Perform stepweight decay
#         p.data.mul_(1 - group['lr'] * group['weight_decay'])
#
#         grad = p.grad
#         state = self.state[p]
#         # State initialization
#         if len(state) == 0:
#           # Exponential moving average of gradient values
#           state['exp_avg'] = torch.zeros_like(p)
#
#         exp_avg = state['exp_avg']
#         beta1, beta2 = group['betas']
#
#         # Weight update
#         update = exp_avg * beta1 + grad * (1 - beta1)
#         p.add_(torch.sign(update), alpha=-group['lr'])
#         # Decay the momentum running average coefficient
#         exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
#
#     return loss
#
# # def calculate_psnr(img1, img2):
# #     # img1 and img2 have range [0, 255]
# #     img1 = img1.astype(np.float64)
# #     img2 = img2.astype(np.float64)
# #     mse = np.mean((img1 - img2) ** 2)
# #     if mse == 0:
# #       return float("inf")
# #     return 20 * math.log10(255.0 / math.sqrt(mse))
#


def calculate_psnr1(img1, img2):
    # compute mse
    # mse = np.mean((img1-img2)**2)
    mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    # compute psnr
    if mse < 1e-10:
        return 100
    psnr1 = 20 * math.log10(255 / math.sqrt(mse))
    return psnr1


def calculate_psnr2(img1, img2):
    mse = np.mean((img1 / 255.0 - img2 / 255.0) ** 2)
    if mse < 1e-10:
        return 100
    psnr2 = 20 * math.log10(1 / math.sqrt(mse))
    return psnr2


def ssim(img1, img2):
  C1 = (0.01 * 255) ** 2
  C2 = (0.03 * 255) ** 2

  img1 = img1.astype(np.float64)
  img2 = img2.astype(np.float64)
  kernel = cv2.getGaussianKernel(11, 1.5)
  window = np.outer(kernel, kernel.transpose())

  mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
  mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
  mu1_sq = mu1 ** 2
  mu2_sq = mu2 ** 2
  mu1_mu2 = mu1 * mu2
  sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
  sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
  sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

  ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
          (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
  )
  return ssim_map.mean()