#!/usr/bin/env python
import os
import platform
import re
from setuptools import setup, find_packages

with open(os.path.join("untool", "_version.py"), "r", encoding="utf-8") as f:
    version_file = f.read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        version = version_match.group(1)
    else:
        raise RuntimeError("无法从_version.py中找到版本信息")

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# 检测系统架构
arch = platform.machine()
if arch == 'x86_64':
    supported_modes = ['pcie']
elif arch == 'aarch64':
    supported_modes = ['soc', 'pcie']
else:
    supported_modes = []

# 基本依赖
install_requires = [
    "numpy<2.0.0",
]

setup(
    name="untool",
    version=version,
    author="wlc952",
    author_email="wlc952@zju.edu.cn",
    description="Union Tool for inference on Sophgo chips",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wlc952/untool-python",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    include_package_data=True,
    install_requires=install_requires,
    entry_points={
        'console_scripts': [
            'untool-info=untool.tools.info:main',
        ],
    },
)