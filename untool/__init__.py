from ._version import __version__
import os

# 设置默认模式
_default_mode = os.environ.get("UNTOOL_MODE", None)

# 导入核心功能
from .bindings.wrapper import (
    untensor_create,
    untensor_destroy,
    runtime_init,
    run,
    llm_init,
    llm_forward_first,
    llm_forward_next,
    # 其他函数...
)

def set_mode(mode):
    """
    设置UnTool运行模式（'soc'或'pcie'）
    
    Args:
        mode: 'soc' 或 'pcie'
    """
    global _default_mode
    if mode not in ['soc', 'pcie']:
        raise ValueError("模式必须是'soc'或'pcie'")
    _default_mode = mode
    os.environ["UNTOOL_MODE"] = mode
    
    # 重新加载底层库
    from .core.platform import load_library
    from .bindings import wrapper
    wrapper._lib = load_library(mode)
    wrapper._setup_functions()