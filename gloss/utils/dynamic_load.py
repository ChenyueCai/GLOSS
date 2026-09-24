# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamic class loading
"""
from importlib import import_module
from typing import Type

def retrieve_class(cls_path: str) -> Type:
    """Dynamically imports a class.

    Args:
        cls_path (str): An absolute module path to a class. E.g.,
            'pkg.module.ClassName'

    Raises:
        ValueError: There is no associated module path

    Returns:
        object: _description_
    """
    cls_path_parts = cls_path.split('.')
    cls_name = cls_path_parts[-1]
    module_path = '.'.join(cls_path_parts[:-1])

    if len(module_path) > 0:
        module = import_module(module_path)
        return getattr(module, cls_name)
    raise ValueError("Dynamic class import requires absolute class path.")

def instantiate_class(cls_path: str, *args, **kwargs) -> object:
    """Dynamically import and instantiate a class.

    Args:
        cls_path (str): An absolute module path to a class. E.g.,
            'pkg.module.ClassName'

    Raises:
        ValueError: There is no associated module path

    Returns:
        object: The instantiated class
    """
    cls_obj = retrieve_class(cls_path)
    return cls_obj(*args, **kwargs)
