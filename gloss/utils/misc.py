# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import re
import shutil
from typing import Any, Optional
import numpy as np

from pathlib import Path
from PIL import Image

import requests
import torch

import hashlib


def todevice(data: Any, device: torch.device, error_on_non_tensor: bool = True) -> Any:
    """Attempt to move any data type onto the given device.

    This method is applied recursively to move containers of
    data onto the specified device.

    Args:
        data (Any): The input data.
        device (torch.device): The target device.
        error_on_non_tensor (bool, optional): Throw an error if a
            non-tensor is attempted to move device. Defaults to True.

    Raises:
        ValueError: If the data type is unsupported, and
            error_on_non_tensor is True.

    Returns:
        Any: The data type now moved to the device.
    """
    if isinstance(data, (tuple, list)):
        return list(
            todevice(d, device, error_on_non_tensor=error_on_non_tensor) for d in data
        )
    elif isinstance(data, dict):
        return {
            k: todevice(v, device, error_on_non_tensor=error_on_non_tensor) for k, v in data.items()
        }
    elif isinstance(data, torch.Tensor):
        return data.to(device=device)
    elif error_on_non_tensor:
        raise ValueError(f"Unsupported data type {type(data)}")
    else:
        return data


def synchronize():
    """
    Synchronize all devices.

    This method checks if the current running environment
    is distributed with a world-size greater than 1.
    If so, we use `torch.distributed.barrier` to synchronize
    all processes.
    """
    if not torch.distributed.is_available():
        return
    if not torch.distributed.is_initialized():
        return

    world_size = torch.distributed.get_world_size()
    if world_size == 1:
        return

    torch.distributed.barrier()


def expand_like(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Expands the input tensor `x` to have the same
    number of dimensions as the `target` tensor.

    Pads `x` with singleton dimensions on the end.

    # Example

    ```
    x = torch.ones(5)
    target = torch.ones(5, 10, 30, 1, 10)

    x = expand_like(x, target)
    print(x.shape) # <- [5, 1, 1, 1, 1]
    ```

    Args:
        x (torch.Tensor): The input tensor to expand.
        target (torch.Tensor): The target tensor whose shape length
            will be matched.

    Returns:
        torch.Tensor: The expanded tensor `x` with trailing singleton
            dimensions.
    """
    while len(x.shape) < len(target.shape):
        x = x[:, None]
    return x


def download_file(url: str, outdir: Optional[os.PathLike] = None) -> os.PathLike:
    """Downloads a file at the given url to the local file system.

    The file will be saved with the filename from the url, within the specified
    directory `outdir`. If the `outdir` does not exist, it will be created.
    If the `outdir` is None, then the working directory is used.

    Args:
        url (str): The url of the target object. Note that this not validated.
        outdir (Optional[os.PathLike], optional): The output directory to save the file.
            Defaults to None.

    Returns:
        os.PathLike: Path to the downloaded file.
    """
    local_filename = url.split('/')[-1]
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        outpath = os.path.join(outdir, local_filename)
    else:
        outpath = local_filename
    if os.path.exists(outpath):
        return outpath
    with requests.get(url, stream=True, timeout=1800) as r:
        with open(outpath, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    return outpath


def reclaim_cuda_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def get_string_hash(name):
    return hashlib.md5(name.encode()).hexdigest()


def to_numpy(tensor:torch.Tensor) -> np.ndarray:
    """_summary_

    Args:
        tensor (torch.Tensor): _description_

    Returns:
        np.ndarray: _description_
    """
    return tensor.cpu().numpy()


def to_tensor(arr:np.ndarray) -> torch.Tensor:
    """_summary_

    Args:
        arr (np.ndarray): _description_

    Returns:
        torch.Tensor: _description_
    """
    return torch.tensor(arr).to(torch.float32).cuda()


def extract_view_numbers(folder):
    """
    Extract numbers from files named like view%04d.png in a given folder.

    Args:
        folder (str): Path to the folder containing PNG files.

    Returns:
        list[int]: Sorted list of extracted numbers.
    """
    pattern = re.compile(r"view(\d{4})\.png$")
    numbers = []

    for fname in os.listdir(folder):
        match = pattern.match(fname)
        if match:
            numbers.append(int(match.group(1)))

    return sorted(numbers)
