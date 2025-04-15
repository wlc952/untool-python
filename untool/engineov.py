import ctypes
import numpy as np
from .bindings.wrapper import (
    unruntime_init,
    unruntime_run,
    unruntime_free,
    unruntime_set_net_stage,
    unruntime_set_input_s2d,
    unruntime_get_io_count,
    unruntime_get_io_tensor,
    convert_to_python,
    type_map)


class EngineOV:
    def __init__(self, model_path="", device_id=0):
        self.model_path = model_path
        self.device_id = device_id
        self.runtime = unruntime_init(model_path, device_id)

    def reset_net_stage(self, net_idx, stage_idx):
        unruntime_set_net_stage(self.runtime, net_idx, stage_idx)
    
    def __str__(self):
        return "EngineOV: model_path={}, device_id={}".format(self.model_path, self.device_id)
    
    def __call__(self, args, to_host=True):
        if isinstance(args, list):
            values = args
        elif isinstance(args, dict):
            values = list(args.values())
        else:
            raise TypeError("args is not list or dict")

        for idx, val in enumerate(values):
            try:
                import torch
                if isinstance(val, torch.Tensor):
                    val = val.cpu().contiguous().numpy()
            except:
                pass

            if not isinstance(val, np.ndarray):
                val = np.asarray(val)
            
            unruntime_set_input_s2d(self.runtime, idx, val.ctypes.data_as(ctypes.c_void_p), val.nbytes)
        
        unruntime_run(self.runtime, to_host)

        out_tensor_ptrs = []
        out_tensor_num = unruntime_get_io_count(self.runtime, b'o')
        for i in range(out_tensor_num):
            out_tensor_ptr = unruntime_get_io_tensor(self.runtime, b'o', i)
            out_tensor_ptrs.append(out_tensor_ptr)

        results = []
        for out_tensor_ptr in out_tensor_ptrs:
            data_ptr = out_tensor_ptr.contents
            dims = int(data_ptr.dims)
            shape = tuple(data_ptr.shape[i] for i in range(dims))
            out_dtype = type_map.get(data_ptr.dtype, np.float32)
            results.append(convert_to_python(data_ptr.data, shape, out_dtype))
        return results
    
    def close(self):
        if self.runtime:
            unruntime_free(self.runtime)
            self.runtime = None
    
    def __del__(self):
        self.close()
