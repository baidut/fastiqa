from fastai.vision.all import *
import torch
import torch.nn.functional as F
import scipy.stats as scs

# only compute based on the first axis (video mos, don't include patch)

@delegates(AccumMetric)
def SRCC(dim_argmax=None, axis=0, nan_policy='propagate', **kwargs):
    "Spearman correlation coefficient for regression problem"
    def spearmanr(x,y=None,**kwargs):
        if x.dim() == 2: x = x[:, 0]
        if y.dim() == 2: y = y[:, 0]
        return scs.spearmanr(x, y,**kwargs)[0]
    return AccumMetric(partial(spearmanr, axis=axis, nan_policy=nan_policy),
                       invert_arg=False, dim_argmax=dim_argmax, flatten=False, **kwargs)

@delegates(AccumMetric)
def LCC(dim_argmax=None, **kwargs):
    "Pearson correlation coefficient for regression problem"
    def pearsonr(x,y):
        if x.dim() == 2: x = x[:, 0]
        if y.dim() == 2: y = y[:, 0]
        return scs.pearsonr(x, y)[0]
    return AccumMetric(pearsonr, invert_arg=False, dim_argmax=dim_argmax, flatten=False, **kwargs)

def DummyLoss(x, y, *args, **kwargs):
    return torch.tensor(np.nan)

def DebugLoss(x, y, *args, **kwargs):
    print(x.shape, y.shape)
    print(x, y)
    return torch.tensor(np.nan)






########################################################

"""
For multi task setting, use task_index = 0, 1, ...
"""

@delegates(AccumMetric)
def PSNR(dim_argmax=None, task_index=None, **kwargs):
    # https://github.com/lychengr3x/Image-Denoising-with-Deep-CNNs/blob/master/src/tutorial.ipynb
    # defined for images ranging in [−1, 1] as
    def psnr(x, y, task_index):
      if task_index is not None:
        x = x[task_index]
        y = y[task_index]
      n = x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3]
      return 10*torch.log10(4*n/(torch.norm(y-x)**2))
    return AccumMetric(lambda x, *y: psnr(x, y, task_index), invert_arg=False, dim_argmax=dim_argmax, flatten=False, **kwargs)

# https://docs.fast.ai/metrics.html#rmse
@delegates(AccumMetric)
def RMSE(dim_argmax=None, task_index=None, **kwargs):
    # https://github.com/lychengr3x/Image-Denoising-with-Deep-CNNs/blob/master/src/tutorial.ipynb
    # defined for images ranging in [−1, 1] as
    def multi_task_rmse(x, y, task_index):
      if task_index is not None:
        x = x[task_index]
        y = y[task_index]
      return torch.sqrt(F.mse_loss(x, y))
    return AccumMetric(lambda x, *y: multi_task_rmse(x, y, task_index), invert_arg=False, dim_argmax=dim_argmax, flatten=False, **kwargs)





"""
# https://www.geeksforgeeks.org/python-peak-signal-to-noise-ratio-psnr/
# numpy
from math import log10, sqrt
import numpy as np
mse = np.mean((y - x) ** 2)
if(mse == 0):  # MSE is zero means no noise is present in the signal .
              # Therefore PSNR have no importance.
    return 100
max_pixel = 255.0
psnr = 20 * log10(max_pixel / sqrt(mse))
"""

"""
# or
import cv2
img1 = cv2.imread('img1.bmp')
img2 = cv2.imread('img2.bmp')
psnr = cv2.PSNR(img1, img2)

"""


# for quality losses: https://github.com/photosynthesis-team/piq
