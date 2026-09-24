# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from typing import get_origin, get_args, Union
from dataclasses import dataclass, fields, asdict
import os
import simple_parsing
import yaml

import gloss.logging


def unwrap_optional(field_type):
    if get_origin(field_type) is Union:
        args = get_args(field_type)
        non_none = [arg for arg in args if arg is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return field_type


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() == 'true':
        return True
    if v.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError("Expected 'True' or 'False'")


def extract_dataclass(prefix, cls, args):
    prefix_dot = f"{prefix}."
    relevant = {
        key[len(prefix_dot):]: value
        for key, value in vars(args).items()
        if key.startswith(prefix_dot) and value is not None
    }
    for key in list(vars(args).keys()):
        if key.startswith(prefix_dot):
            delattr(args, key)
    setattr(args, prefix, cls(**relevant))


class ParserHelper(object):
    """ Helper class to set up arguments from dataclasses that can then also be overridden.
    Note! provides default global_root_dir flag; use it for all path configurations.

    Example:
        @dataclass
        class MyDataConfig:
            path: str
            batch_size: int = 16

        def main(...):
            parser_helper = ParserHelper()
            parser_helper.add_dataclass_flags(MyDataConfig, 'data')
            parser_helper.parser.add_argument('--some_other_flag', action='store_true', default=False)
            args = parser_helper.parse_args()

            # write config
            parser_helper.write_config_yml('/tmp/my_config.yml')

            data_config = args.data  # of type MyDataConfig

        # now can call main script with flags that override anything in data_config, e.g.
        python my_main_script.py --data.path=my_path/to/data --data.batch_size=64

        # you can also re-run script with exact same flags by *only* providing saved config from prior run
        python my_main_script.py --config=/tmp/my_config.yml  # will have batch_size 64

    """
    def __init__(self, description):
        self.parser = simple_parsing.ArgumentParser(description=description,
                                                    add_config_path_arg=True,
                                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument('--global_root_dir', default='', type=str,
                                 help='Global root path to pre-pend to all the config directories; '
                                 'useful for running same configs on cluster and locally.')
        gloss.logging.add_log_level_flag(self.parser)
        self._configurable = set()
        self.args = None

    def parse_args(self, *args, **kwargs):
        self.args = self.parser.parse_args(*args, **kwargs)
        gloss.logging.default_log_setup(self.args.log_level)
        for cls, name in self._configurable:
            extract_dataclass(name, cls, self.args)
        return self.args

    def add_dataclass_flags(self, cls, name):
        for field in fields(cls):
            arg_name = f"--{name}.{field.name}"
            dest_name = f"{name}.{field.name}"
            arg_type = unwrap_optional(field.type)
            if arg_type == list[str]:
                self.parser.add_argument(arg_name, dest=dest_name, nargs='*', type=str)
            elif arg_type == bool:
                self.parser.add_argument(arg_name, dest=dest_name, type=str2bool)
            else:
                self.parser.add_argument(arg_name, dest=dest_name, type=arg_type)
        self._configurable.add((cls, name))

    def get_config_dict(self, args=None):
        if args is None:
            args = self.args
        data = {}
        for _, name in self._configurable:
            val = getattr(args, name, None)
            if val is not None:
                data[name] = asdict(val)
        return data

    def write_config_yml(self, fpath, args=None):
        if args is None:
            args = self.args
        with open(fpath, 'w') as f:
            f.write(yaml.dump(self.get_config_dict(args)))


