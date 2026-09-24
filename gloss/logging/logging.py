# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import glob
import logging
import os
import re
import sys

import kaolin

logger = logging.getLogger(__name__)


def default_log_setup(level=logging.INFO, filename=None, append_count=False):
    """
    Sets up default logging, always logging to stdout as well as
    file, if file is specified. If append_count, will search for
    filename with count and create a unique filename with the next
    int count before the extension.
    E.g., repeated calls to:
      defaultLoggingSetup(filename='log.txt', append_count=True)
    will result in files:
      log00.txt
      log01.txt
      log02.txt ...

    :param level: logging level, e.g. logging.INFO
    :param filename: if to log to file in addition to stdout, set filename
    :param append_count: see above
    :return: count, if append_count, else None
    """
    handlers = [logging.StreamHandler(sys.stdout)]
    count = None
    if filename is not None:
        if append_count:
            count = 0
            directory = os.path.dirname(filename)
            basename, extension = os.path.splitext(os.path.basename(filename))
            pattern = re.compile(r"%s(\d+)%s" % (basename, extension))
            fnames = [
                os.path.basename(x) for x in glob.glob(os.path.join(directory, "%s[0-9]*%s" % (basename, extension)))
            ]
            counts = [int(m.groups()[0]) for m in [pattern.match(f) for f in fnames] if m is not None]
            if len(counts) > 0:
                count = max(counts) + 1
            filename = os.path.join(directory, "%s%02d%s" % (basename, count, extension))
        handlers.append(logging.FileHandler(filename))
    logging.basicConfig(level=level, format="%(asctime)s|%(levelname)8s|%(name)15s| %(message)s", handlers=handlers)
    logger.info("Logging to stdout and %s" % filename)
    logging.getLogger("PIL.PngImagePlugin").setLevel(20)
    return count


def add_log_level_flag(parser):
    parser.add_argument(
        "--log_level",
        action="store",
        type=int,
        default=logging.INFO,
        help="Logging level to use globally, DEBUG: 10, INFO: 20, WARN: 30, ERROR: 40.",
    )


def log_tensor(t, name, use_logger=None, level=logging.DEBUG, print_stats=False, detailed=False):
    use_logger.log(level, kaolin.utils.testing.tensor_info(t, name, print_stats=print_stats, detailed=detailed))


def log_tensor_dict(d, name, use_logger, level=logging.DEBUG, **log_kwargs):
    use_logger.log(level, "Tensor dict %s" % name)
    for k, v in d.items():
        log_tensor(v, str(k), use_logger, level=level, **log_kwargs)
