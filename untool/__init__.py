from ._version import __version__
import os

if os.environ.get("UNTOOL_MODE") is None:
    os.environ["UNTOOL_MODE"] = "soc"

from .bindings.wrapper import (
    # C API
    convert_model_info,
    get_model_info_p,
    find_net_num,
    move_to_device,
    compile_io_addr,
    fill_api_info,
    run_model,
    free_model,
    untensor_create,
    untensor_destroy,
    untensor_set_data,
    untensor_sync,
    untensor_show,
    untensor_s2d_bytes,
    untensor_d2d_bytes_offset,
    untensor_malloc_device,
    untensor_free_device,
    unruntime_init,
    unruntime_get_net_num,
    unruntime_set_net_stage,
    unruntime_get_io_tensor,
    unruntime_get_io_count,
    unruntime_set_input_s2d,
    unruntime_set_input_d2d,
    unruntime_run,
    unruntime_free,
    # utils
    get_lib_path,
    type_map,
    convert_to_ctypes,
    convert_to_python,
    char_point_2_str,
    )

from .enginellm import EngineLLM
from .engineov import EngineOV
from .llmbasemodel import LLMBaseModel, MiniCPMV
from .llmbasepipeline import LLMBasePipeline, MiniCPMVPipeline

__all__ = [
    "__version__",
    # C API
    "convert_model_info",
    "get_model_info_p",
    "find_net_num",
    "move_to_device",
    "compile_io_addr",
    "fill_api_info",
    "run_model",
    "free_model",
    "untensor_create",
    "untensor_destroy",
    "untensor_set_data",
    "untensor_sync",
    "untensor_show",
    "untensor_s2d_bytes",
    "untensor_d2d_bytes_offset",
    "untensor_malloc_device",
    "untensor_free_device",
    "unruntime_init",
    "unruntime_get_net_num",
    "unruntime_set_net_stage",
    "unruntime_get_io_tensor",
    "unruntime_get_io_count",
    "unruntime_set_input_s2d",
    "unruntime_set_input_d2d",
    "unruntime_run",
    "unruntime_free",
    # Python API
    "EngineLLM",
    "EngineOV",
    "LLMBaseModel",
    "LLMBasePipeline",
    "MiniCPMV",
    "MiniCPMVPipeline",
    # utils
    "get_lib_path",
    "type_map",
    "convert_to_ctypes",
    "convert_to_python",
    "char_point_2_str",
]