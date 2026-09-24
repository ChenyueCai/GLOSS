# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import numpy as np
import os
import torch

logger = logging.getLogger(__name__)

def load_part_segmentation(pattern, max_num=20):
    """
    Loads pre-aggregated segmentation from files such as here:

    Example:
        seg, colors = load_part_segmentation('masha_data_kmeans/cluster_out/sea_dragon_0_%02d.npy')
    Args:
        pattern:
        max_num:

    Returns:

    """
    results = []
    seq_ids = {}
    colors = []
    for i in range(1, max_num + 1):
        fname = pattern % i
        if not os.path.exists(fname):
            if i == 1:
                logger.warning(f'Could not find file matching {fname}')
            break
        res = torch.from_numpy(np.load(fname))
        res_final = torch.zeros_like(res)
        uids = torch.unique(res)
        for uidx in range(uids.shape[0]):
            uid = uids[uidx].item()
            if uid not in seq_ids:
                seq_ids[uid] = len(seq_ids) + 1
                colors.append(torch.tensor([(len(seq_ids) % 28) / 28.0, (len(seq_ids) % 9) / 9.0, (len(seq_ids) % 3) /3.0]))
            sid = seq_ids[uid]
            res_final[res == uid] = sid
        results.append(res_final)
    if len(results) == 0:
        return None, None

    return torch.stack(results), torch.stack(colors)