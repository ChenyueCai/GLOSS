# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import re
import os
import shutil
from tqdm import tqdm
import tarfile
from itertools import chain
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from gloss.data.render_webdataset import FILE_SUFFIXES


def create_shard(shard_file, base_dir, view, filenames):
    with tarfile.open(shard_file, "w") as tar:
        for img_file in filenames:
            img_path = os.path.join(base_dir, img_file)
            tar.add(img_path, arcname=f"{view}/{img_file}")


def chunk_list(lst, chunk_size):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def process_view(view, base_dir, target_dir):
    _data_cache_dir = os.path.join(base_dir, view, 'datacache')
    assert os.path.exists(_data_cache_dir)
    image_files = [f for f in os.listdir(_data_cache_dir) if f.endswith(".png")]
    prefix_pattern = re.compile(
        r"^(e-?\d+\.\d+_-?\d+\.\d+_-?\d+\.\d+"
        r"a-?\d+\.\d+_-?\d+\.\d+_-?\d+\.\d+"
        r"u-?\d+\.\d+_-?\d+\.\d+_-?\d+\.\d+"
        r"_f-?\d+\.\d+)"
    )
    grouped = defaultdict(list)

    for filename in image_files:
        match = prefix_pattern.match(filename)
        if match:
            prefix = match.group(1)
            grouped[prefix].append(filename)

    expected_num_channels = max([len(v) for v in grouped.values()])
    valid_samples = [v for v in grouped.values() if len(v) == expected_num_channels]
    chunks = chunk_list(valid_samples, 100)
    num_tar_files = len(valid_samples) // 100
    for i in range(num_tar_files):
        filenames = list(chain.from_iterable(chunks[i]))
        create_shard(os.path.join(target_dir, f"{view}-{i}.tar"), _data_cache_dir, view, filenames)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_base_dir", type=str, required=True)
    parser.add_argument("--target_base_dir", type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=16, help="Number of threads for parallel processing")
    args = parser.parse_args()

    assert os.path.exists(args.data_base_dir)
    if args.target_base_dir:
        os.makedirs(args.target_base_dir, exist_ok=True)

    views = os.listdir(args.data_base_dir)
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(process_view, view, args.data_base_dir, args.target_base_dir)
            for view in views
        ]
        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass

