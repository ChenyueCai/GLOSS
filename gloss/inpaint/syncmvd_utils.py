# Code taken from: https://github.com/LIU-Yuxin/SyncMVD/blob/main/src/pipeline.py
# MIT License
#
# Copyright (c) 2023 LIU-Yuxin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
from diffusers.utils.torch_utils import randn_tensor


# Revert time 0 background to time t to composite with time t foreground
@torch.no_grad()
def composite_rendered_view(scheduler, backgrounds, foregrounds, masks, t):
    composited_images = []
    for i, (background, foreground, mask) in enumerate(zip(backgrounds, foregrounds, masks)):
        if t > 0:
            alphas_cumprod = scheduler.alphas_cumprod[t]
            noise = torch.normal(0, 1, background.shape, device=background.device)
            background = (1 - alphas_cumprod) * noise + alphas_cumprod * background
        composited = foreground * mask + background * (1 - mask)
        composited_images.append(composited)
    composited_tensor = torch.stack(composited_images)
    return composited_tensor
