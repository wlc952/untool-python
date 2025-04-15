from ._version import __version__
import os

# 设置默认模式
if os.environ.get("UNTOOL_MODE") is None:
    os.environ["UNTOOL_MODE"] = "soc"

# 导入核心功能
from .bindings.wrapper import *