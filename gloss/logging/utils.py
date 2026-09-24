# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import logging
import torch
import wandb
import torchvision
from torchvision.transforms import functional as tv_F

logger = logging.getLogger(__name__)


def get_full_class_name(o: object) -> str:
    """Returns a qualified class name

    For example, given an instance of class Bar in module foo,
    this function will return "foo.Bar".

    Args:
        o (object): An object that we want to get the class name of

    Returns:
        str: The name of the class.
    """
    c = o.__class__
    m = c.__module__
    return m + '.' + c.__qualname__


def get_class_logger(o: object) -> logging.Logger:
    """Get a logger with name identifying the class type.

    This function is intended to be used in the __init__ method of a class.
    For example, in module `foo`:

    ```
    from vmfd.logging import get_class_logger
    class Bar:
        def __init__(self):
            self.logger = get_class_logger(self)
            self.logger.info("Created `Bar` instance")
    ```
    This logger will have the name `foo.Bar` attached.

    Args:
        o (object): An instance of the class

    Returns:
        logging.Logger: A logger named using the class.
    """
    return logging.getLogger(get_full_class_name(o))


def to_wandb_image(
        tensor: torch.Tensor,
        rgb_range: float = 255.0,
        max_plot: int = 4
) -> wandb.Image:
    """Convert the given image tensor into a wandb.Image for logging.

    Args:
        tensor (torch.Tensor): Input image tensor of shape [B,C,H,W]. Assumed to be [0,1] bounded.
        rgb_range (float, optional): Output target RGB range (can almost definitely be kept as 255).
            Defaults to 255.0.
        max_plot (int, optional): Max number of images to plot. Defaults to 4.

    Returns:
        wandb.Image: _description_
    """
    normalized = tensor.mul(rgb_range).clip_(0, rgb_range).to(torch.uint8)
    im_tensor = normalized.byte()[:max_plot]
    image_grid = torchvision.utils.make_grid(im_tensor, nrow=1, pad_value=1)
    image_grid = tv_F.to_pil_image(image_grid)
    wandb_image = wandb.Image(image_grid)
    return wandb_image


# taken from: https://stackoverflow.com/questions/1094841/get-human-readable-version-of-file-size
def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def get_model_size(model: torch.nn.Module) -> int:
    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    mem = mem_params + mem_bufs  # in bytes

    return mem


def print_memory_diagnostics_human(device, threshold=10_000, level=logging.DEBUG):
    assert threshold >= 0
    for obj in gc.get_objects():
        try:
            s = -1
            if isinstance(obj, torch.nn.Module):
                s = get_model_size(obj)
            elif torch.is_tensor(obj):
                s = obj.nelement() * obj.element_size()

            if s > threshold:
                logger.log(level, "Tensor {} memory {}".format(type(obj), sizeof_fmt(s)))
        except Exception:
            pass

    if torch.cuda.is_available():
        logger.log(
            level, f"Device {device} memory: {sizeof_fmt(torch.cuda.get_device_properties(device).total_memory)}"
        )


def print_memory_diagnostics(device, level=logging.DEBUG):
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, "data") and torch.is_tensor(obj.data)):
                logger.log(level, "Tensor {} memory {}".format(type(obj), obj.size()))
        except Exception as e:
            pass

    if torch.cuda.is_available():
        logger.log(
            level,
            "Device {} memory: {}GB".format(
                device, torch.cuda.get_device_properties(device).total_memory / 1000000000.0
            ),
        )