import os
import ctypes
import platform
import numpy as np

# 常量定义
MAX_SHAPE_NUM     = 8
MAX_TENSOR_NUM    = 64
MAX_STAGE_NUM     = 64
MAX_CHAR_NUM      = 128
MAX_NET_NUM       = 256

# 初始化库
def get_lib_path(mode=None):
    arch = platform.machine()
    if mode is None:
        if os.environ.get("UNTOOL_MODE"):
            mode = os.environ.get("UNTOOL_MODE")
        else:
            if arch == 'aarch64':
                mode = 'soc'  # aarch64默认使用soc模式
            else:
                mode = 'pcie'  # x86_64默认使用pcie模式
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
    
try:
    mode = os.environ.get("UNTOOL_MODE", "soc")
    lib_path = get_lib_path(mode)
    lib = ctypes.CDLL(lib_path)
except Exception as e:
    print(f"警告: 初始化untool库失败，可能需要手动设置正确的模式: {e}")

# 类型定义
int_point = ctypes.POINTER(ctypes.c_int)
int_      = ctypes.c_int
ulonglong = ctypes.c_ulonglong
cpoint    = ctypes.c_void_p
vpoint    = ctypes.c_void_p
spoint    = ctypes.c_char_p
bool_     = ctypes.c_bool
null_ptr  = ctypes.c_void_p(None)
ref       = lambda x: ctypes.byref(x)

##### ========================== #####
#####    类型转换: Python to C     #####
##### ========================== #####
def make2_c_uint64_list(my_list):
    """
    将 Python 列表转换为 ctypes 的 c_uint64 数组
    """
    return (ctypes.c_uint64 * len(my_list))(*my_list), 

def make2_c_int_list(my_list):
    """
    将 Python 列表转换为 ctypes 的 c_int 数组
    """
    return (ctypes.c_int * len(my_list))(*my_list)

def make_np2c(np_array):
    """
    将 NumPy 数组转换为 ctypes 的 void* 指针，确保数组为连续内存
    """
    if not np_array.flags['CONTIGUOUS']:
        np_array = np.ascontiguousarray(np_array)
    return np_array.ctypes.data_as(ctypes.c_void_p)

def make_np2c_bf16_from_fp32(np_array):
    """
    将 float32 的 NumPy 数组转换为 bfloat16 表示，返回值为一个元组：
    (ctypes void* 指针, bf16_array 数组)
    """
    if not np_array.flags['CONTIGUOUS']:
        np_array = np.ascontiguousarray(np_array)
    if np_array.dtype != np.float32:
        np_array = np_array.astype(np.float32)
    # 将 float32 数据以 uint32 方式查看
    np_array_uint32 = np_array.view(np.uint32)
    # 右移 16 位转换为 bfloat16，并转换为 uint16 类型
    bf16_array = (np_array_uint32 >> 16).astype(np.uint16)
    return bf16_array.ctypes.data_as(ctypes.c_void_p), bf16_array

def str2char_point(string):
    """
    将 Python 字符串转换为 ctypes 的 char* 指针
    """
    return ctypes.c_char_p(string.encode('utf-8'))

def convert_to_ctypes(data, to_bf16=False):
    """
    通用数据转换函数，将多种类型数据转换为 ctypes 指针：
    
    - 对于 Python 列表：
      返回 convert_to_ctypes(np.array(data), list_target=list_target, np_array_target=np_array_target)
      
    - 对于字符串：
      返回 (char* 指针, 数据字节数, 字符串)。

    - 对于 NumPy 数组：
      当 to_bf16 为 True 时，将 float32 数组转换为 bfloat16 表示，
      返回 (指针, 数据字节数, array)
      
    - 对于标量 (np.generic, int, float)：
      返回 (byref 指针, 数据字节数, 数据)
    """
    if isinstance(data, list):
        try:
            array_list = np.asarray(data)(data)
            return convert_to_ctypes(array_list, to_bf16=to_bf16)
        except Exception as e:
            raise ValueError(f"列表转换失败: {e}")

    if isinstance(data, str):
        ptr = str2char_point(data)
        nbytes = len(data.encode('utf-8'))
        return ptr, nbytes, data

    if isinstance(data, np.ndarray):
        if to_bf16:
            ptr, bf16_array = make_np2c_bf16_from_fp32(data)
            nbytes = bf16_array.nbytes
            return ptr, nbytes, bf16_array
        else:
            ptr = make_np2c(data)
            nbytes = data.nbytes
            return ptr, nbytes, data

    if isinstance(data, (np.generic, int, float)):
        dtype = type(data)
        if dtype in dtype_ctype_map:
            c_data = dtype_ctype_map[dtype](data)
        else:
            c_data = ctypes.c_float(data)
        ptr = ctypes.byref(c_data)
        nbytes = ctypes.sizeof(c_data)
        return ptr, nbytes, c_data

    raise TypeError(f"不支持的数据类型: {type(data)}")

##### ========================== #####
#####    类型转换: C to Python     #####
##### ========================== #####
data_type = {
    np.float32: 0,
    np.float16: 1,
    np.int8:    2,
    np.uint8:   3,
    np.int16:   4,
    np.int32:   6,
    np.dtype(np.float32): 0,
    np.dtype(np.float16): 1,
    np.dtype(np.int8):    2,
    np.dtype(np.uint8):   3,
    np.dtype(np.int16):   4,
    np.dtype(np.int32):   6,
}

type_map = {
    0: np.float32,
    1: np.float16,
    2: np.int8,
    3: np.uint8,
    4: np.int16,
    6: np.int32,
}

dtype_ctype_map = {
    int:        ctypes.c_int,
    float:      ctypes.c_float,
    np.float32: ctypes.c_float,
    np.float16: ctypes.c_uint16,
    np.int8:    ctypes.c_int8,
    np.uint8:   ctypes.c_uint8,
    np.int16:   ctypes.c_int16,
    np.uint16:  ctypes.c_uint16,
    np.int32:   ctypes.c_int32,
    np.uint32:  ctypes.c_uint32,
    np.int64:   ctypes.c_int64,
    np.uint64:  ctypes.c_uint64
}

def make_c2np(data_ptr, shape, dtype):
    """
    将 ctypes 指针转换回指定形状和数据类型的 NumPy 数组
    """
    num = np.prod(shape)
    array_type = ctypes.cast(data_ptr, ctypes.POINTER(dtype_ctype_map[dtype]))
    np_array = np.ctypeslib.as_array(array_type, shape=(num,))
    return np_array.view(dtype=dtype).reshape(shape)

def make_c2np_fp32_from_bf16(data_ptr, shape):
    """
    将包含 bfloat16 数据的 ctypes 指针转换为 float32 的 NumPy 数组
    """
    num = np.prod(shape)
    array_type = ctypes.cast(data_ptr, ctypes.POINTER(ctypes.c_uint16))
    raw_array = np.ctypeslib.as_array(array_type, shape=(num,))
    fp32_array = (raw_array.astype(np.uint32) << 16).view(np.float32)
    return fp32_array.reshape(shape)


def char_point_2_str(char_point):
    """
    将 ctypes 的 char* 指针转换为 Python 字符串
    """
    return ctypes.string_at(char_point).decode('utf-8')

def convert_to_python(data, shape=None, dtype=None, bf16=False):
    """
    将 ctypes 指针转换为 Python 对象：
      - 如果提供 shape 和 dtype，则视为数组转换，返回 NumPy 数组；
        如果 bf16 为 True，则认为数据以 bfloat16 形式存储，使用 make_c2np_fp32_from_bf16 将其转换为 float32 的 NumPy 数组。
      - 如果不提供 shape 和 dtype，则视为标量转换，返回对应的 Python 对象（通过指针内容）。

    参数:
      data: ctypes 指针
      shape: 数组的形状（可选），若为 None，则认为 data 指向标量数据
      dtype: 数组的数据类型（可选），例如 np.float32、np.int32 等，与 shape 同时提供时用于数组转换
      bf16: 布尔值，指示是否将 bfloat16 数据转换为 float32 数组（仅在 shape 和 dtype 被提供时有效）

    返回:
      对应的 Python 对象（NumPy 数组或标量）
    """
    try:
        if shape is not None and dtype is not None:
            if bf16:
                return make_c2np_fp32_from_bf16(data, shape)
            else:
                return make_c2np(data, shape, dtype)
        return data.contents
    except Exception as e:
        raise ValueError("无法转换数据为 Python 对象，可能缺少必要的元数据信息（例如 shape, dtype）") from e
