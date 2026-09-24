# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A simple performance timer.
"""

import time
import torch


class QuickTimer(object):
    """
    Extremely simple timer object; likely replaceable by other python utilities.
    """

    __singleton__ = None

    @staticmethod
    def Singleton():
        if QuickTimer.__singleton__ is None:
            QuickTimer.__singleton__ = QuickTimer()
        return QuickTimer.__singleton__

    def __init__(self):
        self.timers = {}
        self.lastKey = None

    @staticmethod
    def clear():
        QuickTimer.Singleton()._clear()

    @staticmethod
    def start(key):
        QuickTimer.Singleton()._start(key)

    @staticmethod
    def stop(key=None, block_cuda: bool = False):
        QuickTimer.Singleton()._stop(key, block_cuda)

    @staticmethod
    def summary(sort_by_time=True):
        return QuickTimer.Singleton()._summary(sort_by_time)

    def _clear(self):
        self.timers = {}
        self.lastKey = None

    def _start(self, key):
        if key not in self.timers:
            self.timers[key] = {"total": 0, "count": 0}
        self.timers[key]["start"] = time.time()  # TODO: how is this different from perf_counter()?
        self.timers[key]["count"] += 1
        self.lastKey = key

    def _stop(self, key, block_cuda: bool):
        if block_cuda:
            torch.cuda.synchronize()

        if not key:
            key = self.lastKey

        if self.timers[key]["start"] is not None:
            self.timers[key]["total"] += time.time() - self.timers[key]["start"]
            self.timers[key]["start"] = None

    def _summary(self, sort_by_time):
        summ = [
            (self.timers[k]["total"], k, self.timers[k]["total"] / self.timers[k]["count"])
            for k in sorted(self.timers.keys())
        ]
        if sort_by_time:
            summ.sort()
        return "\n".join(["TIMING %s %0.5f -- AVE %0.5f" % (x[1], x[0], x[2]) for x in summ])