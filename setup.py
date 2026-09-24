# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from setuptools import setup, find_packages

with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = f.readlines()


setup(
    name="gloss",
    version="0.0.1",
    author="Maria Shugrina, James Lucas, Chenyue Cai",
    author_email="mshugrina@nvidia.com",
    description="Material superresolution for 3D meshes.",
    long_description_content_type="text/markdown",
    # gloss_interactive has no __init__.py, so find_packages() misses it; list it explicitly.
    packages=find_packages(include=["gloss", "gloss.*"]) + ["gloss_interactive"],
    package_data={"gloss_interactive": ["asset/*.yaml"]},
    python_requires=">=3.9",
    # install_requires=requirements,  #Note: much more reliable to install manually!
    # ext_modules=get_extensions(),
    # cmdclass={"build_ext": cpp_extension.BuildExtension},
    test_suite="tests",
    classifiers=["Operating System :: OS Independent"],
)
