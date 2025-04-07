import platform
import os
import ctypes
import sys

def get_architecture():
    """获取当前系统架构"""
    return platform.machine()

def get_lib_path(mode=None):
    """
    根据系统架构和指定模式获取适合的库路径
    
    Args:
        mode: 'soc' 或 'pcie'，如果为None则尝试自动检测
    
    Returns:
        libuntool.so的完整路径
    """
    arch = get_architecture()
    
    # 检查是否安装了特定模式的包
    if mode is None:
        # 尝试自动检测模式
        if os.environ.get("UNTOOL_MODE"):
            mode = os.environ.get("UNTOOL_MODE")
        else:
            # 默认模式选择策略
            if arch == 'aarch64':
                mode = 'soc'  # aarch64默认使用soc模式
            else:
                mode = 'pcie'  # x86_64默认使用pcie模式
    
    # 验证模式与架构兼容性
    if arch == 'x86_64' and mode == 'soc':
        raise RuntimeError("x86_64平台不支持SOC模式")
    
    # 查找库的位置
    # 1. 先检查环境变量
    if os.environ.get("UNTOOL_LIB_PATH"):
        return os.environ.get("UNTOOL_LIB_PATH")
    
    # 2. 再检查包内预编译库
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lib_path = os.path.join(package_dir, 'libs', arch, mode, 'libuntool.so')
    if os.path.exists(lib_path):
        return lib_path
    
    # 3. 最后尝试系统安装路径
    system_lib_path = "/opt/untool/lib/libuntool.so"
    if os.path.exists(system_lib_path):
        return system_lib_path
    
    raise FileNotFoundError(f"找不到适用于 {arch}/{mode} 的libuntool.so库")

def load_library(mode=None):
    """加载libuntool.so库"""
    lib_path = get_lib_path(mode)
    try:
        return ctypes.CDLL(lib_path)
    except OSError as e:
        raise OSError(f"加载库文件'{lib_path}'失败: {e}")