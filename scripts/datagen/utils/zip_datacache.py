# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import zipfile
import shutil
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def create_uncompressed_zip(zip_path, source_dir, remove_source=False):
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_STORED) as zipf:
        for foldername, _, filenames in os.walk(source_dir):
            for filename in filenames:
                file_path = os.path.join(foldername, filename)
                arcname = os.path.relpath(file_path, source_dir)
                zipf.write(file_path, arcname)
    if remove_source:
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            if zipf.testzip() is None:
                shutil.rmtree(source_dir)
            else:
                raise Exception("Zip verification failed. Original files not deleted.")


def process_view(view, base_dir, target_dir, remove_dir):
    _data_cache_dir = os.path.join(base_dir, view, 'datacache')
    assert os.path.exists(_data_cache_dir)
    output_dir = None
    if target_dir is not None:
        output_dir = os.path.join(target_dir, view)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "datacache.zip")
    else:
        output_file = f"{_data_cache_dir}.zip"
    create_uncompressed_zip(output_file, _data_cache_dir, remove_dir)

    # Optional _ref dir
    _ref_data_cache_dir = _data_cache_dir + "_ref"
    if os.path.exists(_ref_data_cache_dir):
        if output_dir:
            output_file = os.path.join(output_dir, "datacache_ref.zip")
        else:
            output_file = f"{_ref_data_cache_dir}.zip"
        create_uncompressed_zip(output_file, _ref_data_cache_dir, remove_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_base_dir", type=str, required=True)
    parser.add_argument("--target_base_dir", type=str, default=None)
    parser.add_argument('--multi_view', action='store_true', default=False)
    parser.add_argument('--remove_dir', action='store_true', default=False)
    parser.add_argument('--num_workers', type=int, default=16, help="Number of threads for parallel processing")
    args = parser.parse_args()

    assert os.path.exists(args.data_base_dir)
    if args.target_base_dir:
        os.makedirs(args.target_base_dir, exist_ok=True)

    if args.multi_view:
        views = os.listdir(args.data_base_dir)
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [
                executor.submit(process_view, view, args.data_base_dir, args.target_base_dir, args.remove_dir)
                for view in views
            ]
            for _ in tqdm(as_completed(futures), total=len(futures)):
                pass
    else:
        args.data_base_dir = args.data_base_dir.rstrip("/")
        if args.target_base_dir:
            output_file = os.path.join(args.target_base_dir, os.path.basename(args.data_base_dir) + ".zip")
        else:
            output_file = f"{args.data_base_dir}.zip"
        create_uncompressed_zip(output_file, args.data_base_dir, args.remove_dir)
