import ctypes
import numpy as np
import os
from ..core.platform import load_library

# 常量定义
MAX_DIMS          = 6
MAX_SHAPE_NUM     = 6
MAX_TENSOR_NUM    = 64
MAX_CMD_GROUP_NUM = 64
MAX_STAGE_NUM     = 64
MAX_CHAR_NUM      = 128
MAX_NET_NUM       = 256

# 初始化库
# 初始化库和设置函数
try:
    _lib = load_library()
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
      - 如果 data 是 ctypes 的 char* 指针，则转换为字符串。

    参数:
      data: ctypes 指针
      shape: 数组的形状（可选），若为 None，则认为 data 指向标量数据
      dtype: 数组的数据类型（可选），例如 np.float32、np.int32 等，与 shape 同时提供时用于数组转换
      bf16: 布尔值，指示是否将 bfloat16 数据转换为 float32 数组（仅在 shape 和 dtype 被提供时有效）

    返回:
      对应的 Python 对象（NumPy 数组、标量或字符串）
    """
    if isinstance(data, ctypes.c_char_p):
        return char_point_2_str(data)
    
    if shape is not None and dtype is not None:
        if bf16:
            return make_c2np_fp32_from_bf16(data, shape)
        else:
            return make_c2np(data, shape, dtype)
    
    try:
        return data.contents
    except Exception as e:
        raise ValueError("无法转换数据为 Python 对象，可能缺少必要的元数据信息（例如 shape, dtype）") from e

class UnTensor(ctypes.Structure):
    _fields_ = [
        ('name', ctypes.c_char * MAX_CHAR_NUM),
        ('dtype', ctypes.c_int),
        ('size', ctypes.c_size_t),
        ('dims', ctypes.c_size_t),
        ('shape', ctypes.c_uint64 * MAX_DIMS),
        ('data', ctypes.c_void_p),
        ('is_malloc_host', ctypes.c_bool),
        ('is_have_data', ctypes.c_bool),
        ('device_id', ctypes.c_int),
        ('bm_handle', ctypes.c_void_p),
        ('is_in_device', ctypes.c_bool),
        ('addr', ctypes.c_uint64),
        ('device_start', ctypes.c_uint64),
        ('device_size', ctypes.c_uint64),
        ('offset', ctypes.c_uint64),
        ('dmabuf_fd', ctypes.c_int),
        ('reserved', ctypes.c_uint),
        ('rawflags', ctypes.c_uint),
    ]


lib.untensor_create.restype  = ctypes.POINTER(UnTensor)
def untensor_create() -> ctypes.POINTER(UnTensor):
    """
    untensor untensor_create();
    """
    return lib.untensor_create()

lib.untensor_copy.restype  = ctypes.POINTER(UnTensor)
lib.untensor_copy.argtypes = [ctypes.POINTER(UnTensor), ctypes.c_bool]
def untensor_copy(src, copy_host_data) -> ctypes.POINTER(UnTensor):
    """
    untensor untensor_copy(untensor src, bool copy_host_data);
    :param src: 	ctypes.POINTER(UnTensor)
    :param copy_host_data: 	ctypes.c_bool
    """
    return lib.untensor_copy(src, copy_host_data)

lib.untensor_destroy.restype  = None
lib.untensor_destroy.argtypes = [ctypes.POINTER(UnTensor)]
def untensor_destroy(tensor) -> None:
    """
    void untensor_destroy(untensor tensor);
    :param tensor: 	ctypes.POINTER(UnTensor)
    """
    return lib.untensor_destroy(tensor)

lib.untensor_set_data.restype  = None
lib.untensor_set_data.argtypes = [ctypes.POINTER(UnTensor), ctypes.c_void_p, ctypes.c_size_t, ctypes.c_bool]
def untensor_set_data(tensor, data, size, copy) -> None:
    """
    void untensor_set_data(untensor tensor, void* data, size_t size, bool copy);
    :param tensor: 	ctypes.POINTER(UnTensor)
    :param data: 	ctypes.c_void_p
    :param size: 	ctypes.c_size_t
    :param copy: 	ctypes.c_bool
    """
    return lib.untensor_set_data(tensor, data, ctypes.c_size_t(size), copy)

lib.untensor_sync.restype  = None
lib.untensor_sync.argtypes = [ctypes.POINTER(UnTensor), ctypes.c_bool, ctypes.c_bool]
def untensor_sync(tensor, to_device, force) -> None:
    """
    void untensor_sync(untensor tensor, bool to_device, bool force);
    :param tensor: 	ctypes.POINTER(UnTensor)
    :param to_device: 	ctypes.c_bool
    :param force: 	ctypes.c_bool
    """
    return lib.untensor_sync(tensor, to_device, force)

lib.untensor_show.restype  = None
lib.untensor_show.argtypes = [ctypes.POINTER(UnTensor), ctypes.c_int, ctypes.c_int, ctypes.c_char]
def untensor_show(tensor, start, len, location) -> None:
    """
    void untensor_show(untensor tensor, int start, int len, char location);
    :param tensor: 	ctypes.POINTER(UnTensor)
    :param start: 	ctypes.c_int
    :param len: 	ctypes.c_int
    :param location: 	ctypes.c_char
    """
    return lib.untensor_show(tensor, ctypes.c_int(start), ctypes.c_int(len), location)

lib.runtime_init.restype  = ctypes.c_void_p
lib.runtime_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
def runtime_init(bmodel_path, device_id) -> ctypes.c_void_p:
    """
    unruntime runtime_init(const char* bmodel_path, int device_id);
    :param bmodel_path: 	ctypes.c_char_p
    :param device_id: 	ctypes.c_int
    """
    return lib.runtime_init(str2char_point(bmodel_path), ctypes.c_int(device_id))

lib.get_net_num.restype  = ctypes.c_size_t
lib.get_net_num.argtypes = [ctypes.c_void_p, ctypes.c_char]
def get_net_num(runtime, *net_name) -> ctypes.c_size_t:
    """
    size_t get_net_num(unruntime runtime, const char *net_name);
    :param runtime: 	ctypes.c_void_p
    :param *net_name: 	ctypes.c_char
    """
    return lib.get_net_num(runtime, *net_name)

lib.set_net_stage.restype  = None
lib.set_net_stage.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
def set_net_stage(runtime, net_idx, stage_idx) -> None:
    """
    void set_net_stage(unruntime runtime, size_t net_idx, size_t stage_idx);
    :param runtime: 	ctypes.c_void_p
    :param net_idx: 	ctypes.c_size_t
    :param stage_idx: 	ctypes.c_size_t
    """
    return lib.set_net_stage(runtime, ctypes.c_size_t(net_idx), ctypes.c_size_t(stage_idx))

lib.get_io_tensor.restype  = ctypes.POINTER(UnTensor)
lib.get_io_tensor.argtypes = [ctypes.c_void_p, ctypes.c_char, ctypes.c_size_t]
def get_io_tensor(runtime, io_type, tensor_idx) -> ctypes.POINTER(UnTensor):
    """
    untensor get_io_tensor(unruntime runtime, char io_type, size_t tensor_idx);
    :param runtime: 	ctypes.c_void_p
    :param io_type: 	ctypes.c_char
    :param tensor_idx: 	ctypes.c_size_t
    """
    return lib.get_io_tensor(runtime, io_type, ctypes.c_size_t(tensor_idx))

lib.get_io_count.restype  = ctypes.c_size_t
lib.get_io_count.argtypes = [ctypes.c_void_p, ctypes.c_char]
def get_io_count(runtime, io_type) -> ctypes.c_size_t:
    """
    size_t get_io_count(unruntime runtime, char io_type);
    :param runtime: 	ctypes.c_void_p
    :param io_type: 	ctypes.c_char
    """
    return lib.get_io_count(runtime, io_type)

lib.set_input_s2d.restype  = None
lib.set_input_s2d.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
def set_input_s2d(runtime, tensor_idx, data, size) -> None:
    """
    void set_input_s2d(unruntime runtime, size_t tensor_idx, void* data, size_t size);
    :param runtime: 	ctypes.c_void_p
    :param tensor_idx: 	ctypes.c_size_t
    :param data: 	ctypes.c_void_p
    :param size: 	ctypes.c_size_t
    """
    return lib.set_input_s2d(runtime, ctypes.c_size_t(tensor_idx), data, ctypes.c_size_t(size))

lib.run.restype  = None
lib.run.argtypes = [ctypes.c_void_p, ctypes.c_bool]
def run(runtime, output_to_host) -> None:
    """
    void run(unruntime runtime, bool output_to_host);
    :param runtime: 	ctypes.c_void_p
    :param output_to_host: 	ctypes.c_bool
    """
    return lib.run(runtime, output_to_host)

lib.free_runtime.restype  = None
lib.free_runtime.argtypes = [ctypes.c_void_p]
def free_runtime(runtime) -> None:
    """
    void free_runtime(unruntime runtime);
    :param runtime: 	ctypes.c_void_p
    """
    return lib.free_runtime(runtime)

lib.llm_init.restype  = ctypes.c_void_p
lib.llm_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
def llm_init(model_path, device_id) -> ctypes.c_void_p:
    """
    llmbase llm_init(const char* model_path, int device_id);
    :param model_path: 	ctypes.c_char_p
    :param device_id: 	ctypes.c_int
    """
    return lib.llm_init(str2char_point(model_path), ctypes.c_int(device_id))

lib.llm_forward_first.restype  = ctypes.c_int
lib.llm_forward_first.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_size_t]
def llm_forward_first(llm, tokens, token_len) -> ctypes.c_int:
    """
    int llm_forward_first(llmbase llm, int* tokens, size_t token_len);
    :param llm: 	ctypes.c_void_p
    :param tokens: 	ctypes.POINTER(ctypes.c_int)
    :param token_len: 	ctypes.c_size_t
    """
    return lib.llm_forward_first(llm, make2_c_int_list(tokens), ctypes.c_size_t(token_len))

lib.llm_forward_next.restype  = ctypes.c_int
lib.llm_forward_next.argtypes = [ctypes.c_void_p]
def llm_forward_next(llm) -> ctypes.c_int:
    """
    int llm_forward_next(llmbase llm);
    :param llm: 	ctypes.c_void_p
    """
    return lib.llm_forward_next(llm)

lib.llm_free.restype  = None
lib.llm_free.argtypes = [ctypes.c_void_p]
def llm_free(llm) -> None:
    """
    void llm_free(llmbase llm);
    :param llm: 	ctypes.c_void_p
    """
    return lib.llm_free(llm)

lib.llm_get_seq_len.restype  = ctypes.c_size_t
lib.llm_get_seq_len.argtypes = [ctypes.c_void_p]
def llm_get_seq_len(llm) -> ctypes.c_size_t:
    """
    size_t llm_get_seq_len(llmbase llm);
    :param llm: 	ctypes.c_void_p
    """
    return lib.llm_get_seq_len(llm)

lib.llm_get_token_len.restype  = ctypes.c_size_t
lib.llm_get_token_len.argtypes = [ctypes.c_void_p]
def llm_get_token_len(llm) -> ctypes.c_size_t:
    """
    size_t llm_get_token_len(llmbase llm);
    :param llm: 	ctypes.c_void_p
    """
    return lib.llm_get_token_len(llm)

lib.llm_get_token.restype  = ctypes.c_int
lib.llm_get_token.argtypes = [ctypes.c_void_p]
def llm_get_token(llm) -> ctypes.c_int:
    """
    int llm_get_token(llmbase llm);
    :param llm: 	ctypes.c_void_p
    """
    return lib.llm_get_token(llm)

lib.print_data_by_fp32.restype  = None
lib.print_data_by_fp32.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
def print_data_by_fp32(data, size, dtype, start, len) -> None:
    """
    void print_data_by_fp32(void* data, int size, int dtype, int start, int len);
    :param data: 	ctypes.c_void_p
    :param size: 	ctypes.c_int
    :param dtype: 	ctypes.c_int
    :param start: 	ctypes.c_int
    :param len: 	ctypes.c_int
    """
    return lib.print_data_by_fp32(data, ctypes.c_int(size), ctypes.c_int(dtype), ctypes.c_int(start), ctypes.c_int(len))

lib.data_convert_to_fp32.restype  = None
lib.data_convert_to_fp32.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
def data_convert_to_fp32(src, target, dtype, size) -> None:
    """
    void data_convert_to_fp32(void* src, void* target, int dtype, size_t size);
    :param src: 	ctypes.c_void_p
    :param target: 	ctypes.c_void_p
    :param dtype: 	ctypes.c_int
    :param size: 	ctypes.c_size_t
    """
    return lib.data_convert_to_fp32(src, target, ctypes.c_int(dtype), ctypes.c_size_t(size))

lib.convert_to_fp32.restype  = ctypes.c_float
lib.convert_to_fp32.argtypes = [ctypes.c_void_p, ctypes.c_int]
def convert_to_fp32(source, dtype) -> ctypes.c_float:
    """
    float convert_to_fp32(void* source, int dtype);
    :param source: 	ctypes.c_void_p
    :param dtype: 	ctypes.c_int
    """
    return lib.convert_to_fp32(source, ctypes.c_int(dtype))
version='2025-04-07-11-46-16'
