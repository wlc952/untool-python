import ctypes
from .bindings.wrapper import (
    get_model_info_p,
    convert_model_info,
    move_to_device,
    compile_io_addr,
    fill_api_info,
    run_model,
    free_model,
    find_net_num,
    get_first_mask_ptr,
    get_next_mask_ptr,
    free_mask_ptr,
    untensor_create,
    untensor_destroy,
    untensor_sync,
    untensor_s2d_bytes,
    untensor_d2d_bytes_offset,)


class LLMBaseModel:
    def __init__(self, model_path, device_id=0, quant_type='bf16', generation_mode='greedy'):
        self.generation_mode = generation_mode
        q = quant_type.lower()
        if q in ['bf16', 'bfloat16']:
            self.is_bf16 = True
        elif q in ['f16', 'fp16', 'float16']:
            self.is_bf16 = False
        else:
            raise ValueError(f"Unsupported quant_type: {quant_type}. Use `model_tool --info xxx.bmodel` to check the quant_type.")
        
        self.device_id = device_id
        self.model_info_p = get_model_info_p(model_path, device_id)
        self._init_device()

        self.model_info_c_p = convert_model_info(self.model_info_p)
        self.bm_handle = self.model_info_c_p.contents.bm_handle
        self._init_idx()
        self._init_layers()
        self._init_tensors()

    def _init_device(self):
        move_to_device(self.model_info_p)
        compile_io_addr(self.model_info_p)
        fill_api_info(self.model_info_p)
    
    def _init_idx(self):
        self.embedding_idx = find_net_num(self.model_info_p, 'embedding')
        self.embedding_cache_idx = find_net_num(self.model_info_p, 'embedding_cache')
        self.lm_head_idx = find_net_num(self.model_info_p, 'lm_head')
        self.greedy_head_idx = find_net_num(self.model_info_p, 'greedy_head')
        self.penalty_sample_head_idx = find_net_num(self.model_info_p, 'penalty_sample_head')
        self.SEQLEN = self.model_info_c_p.contents.nets[self.embedding_idx].stages[0].input_tensor[0].shape[1]
    
    def _init_layers(self):
        self.num_layers = 0
        self.block_ids = []
        self.block_cache_ids = []
        while True:
            block_id = find_net_num(self.model_info_p, f'block_{self.num_layers}')
            block_cache_id = find_net_num(self.model_info_p, f'block_cache_{self.num_layers}')
            if block_id == -1 or block_cache_id == -1:
                break
            self.block_ids.append(block_id)
            self.block_cache_ids.append(block_cache_id)
            self.num_layers += 1

    def _init_tensors(self):
        net_num = self.model_info_c_p.contents.net_num
        self.input_tensors = []
        self.output_tensors = []
        for i in range(net_num):
            net_x_input_tensors = []
            net_x_output_tensors = []
            stage_info = self.model_info_c_p.contents.nets[i].stages[0]
            io_alone = stage_info.io_alone
            if io_alone:
                io_device = stage_info.io_device
            else:
                io_device = self.model_info_c_p.contents.neuron_device
            input_num = stage_info.input_num
            output_num = stage_info.output_num
            for i in range(input_num):
                src_tensor = stage_info.input_tensor[i]
                dst_tensor = untensor_create();
                dst_tensor.contents.name = src_tensor.name
                dst_tensor.contents.dtype = src_tensor.data_type
                dst_tensor.contents.dims = src_tensor.dims
                dst_tensor.contents.shape = src_tensor.shape
                dst_tensor.contents.size = src_tensor.size
                dst_tensor.contents.device_id = self.device_id
                dst_tensor.contents.bm_handle = self.bm_handle
                dst_tensor.contents.is_in_device = True
                dst_tensor.contents.addr = stage_info.input_tensor_global_addr[i]
                dst_tensor.contents.device_start = io_device.addr
                dst_tensor.contents.device_size = io_device.size
                dst_tensor.contents.dmabuf_fd = io_device.dmabuf_fd
                dst_tensor.contents.reserved = io_device.reserved;
                dst_tensor.contents.rawflags = io_device.rawflags;
                dst_tensor.contents.offset = dst_tensor.contents.addr - io_device.addr
                net_x_input_tensors.append(dst_tensor)
            for i in range(output_num):
                src_tensor = stage_info.output_tensor[i]
                dst_tensor = untensor_create();
                dst_tensor.contents.name = src_tensor.name
                dst_tensor.contents.dtype = src_tensor.data_type
                dst_tensor.contents.dims = src_tensor.dims
                dst_tensor.contents.shape = src_tensor.shape
                dst_tensor.contents.size = src_tensor.size
                dst_tensor.contents.device_id = self.device_id
                dst_tensor.contents.bm_handle = self.bm_handle
                dst_tensor.contents.is_in_device = True
                dst_tensor.contents.addr = stage_info.output_tensor_global_addr[i]
                dst_tensor.contents.device_start = io_device.addr
                dst_tensor.contents.device_size = io_device.size
                dst_tensor.contents.dmabuf_fd = io_device.dmabuf_fd
                dst_tensor.contents.reserved = io_device.reserved;
                dst_tensor.contents.rawflags = io_device.rawflags;
                dst_tensor.contents.offset = dst_tensor.contents.addr - io_device.addr
                net_x_output_tensors.append(dst_tensor)
            self.input_tensors.append(net_x_input_tensors)
            self.output_tensors.append(net_x_output_tensors)
            
    def s2d_bytes(self, src, data, byte_size):
        untensor_s2d_bytes(src, data, byte_size)
    
    def d2d_bytes_offset(self, dst, src, dst_offset, src_offset, byte_size):
        untensor_d2d_bytes_offset(self.bm_handle, dst, src, dst_offset, src_offset, byte_size)

    def forward_first(self, input_ids):
        self.token_length = len(input_ids)
        # 参数1 input_ids
        input_ids = input_ids + [0] * (self.SEQLEN - self.token_length)
        c_input_ids = (ctypes.c_int * len(input_ids))(*input_ids)
        input_ids_ptr = ctypes.cast(c_input_ids, ctypes.c_void_p)

        # 参数2 position_id
        position_id = [0] * self.SEQLEN
        for i in range(self.token_length):
            position_id[i] = i
        c_position_id = (ctypes.c_int * len(position_id))(*position_id)
        position_id_ptr = ctypes.cast(c_position_id, ctypes.c_void_p)

        # 参数3 attention_mask
        attn_mask_ptr = get_first_mask_ptr(self.SEQLEN, self.token_length, self.is_bf16)

        # 网络1 embedding
        in_tensor = self.input_tensors[self.embedding_idx][0]
        out_tensor = self.output_tensors[self.embedding_idx][0]

        self.s2d_bytes(in_tensor, input_ids_ptr, 4 * self.SEQLEN)
        run_model(self.model_info_p, self.embedding_idx, 0)

        # 网络2 attention blocks
        for i in range(self.num_layers):
            net_id = self.block_ids[i]
            self.d2d_bytes_offset(self.input_tensors[net_id][0], out_tensor, 0, 0, out_tensor.contents.size)
            if i == 0:
                self.s2d_bytes(self.input_tensors[net_id][1], position_id_ptr, 4 * self.SEQLEN)
                self.s2d_bytes(self.input_tensors[net_id][2], attn_mask_ptr, 2 * self.SEQLEN * self.SEQLEN)
            run_model(self.model_info_p, net_id, 0)
            out_tensor = self.output_tensors[net_id][0]
            cache_id = self.block_cache_ids[i]
            self.d2d_bytes_offset(self.input_tensors[cache_id][3], self.output_tensors[net_id][1], 0, 0, self.output_tensors[net_id][1].contents.size)
            self.d2d_bytes_offset(self.input_tensors[cache_id][4], self.output_tensors[net_id][2], 0, 0, self.output_tensors[net_id][2].contents.size)
        
        # 网络2 lm_head
        bytes_size = out_tensor.contents.size // self.SEQLEN
        src_offset = bytes_size * (self.token_length - 1)

        token_tensor = self.output_tensors[self.lm_head_idx][0]
        self.d2d_bytes_offset(self.input_tensors[self.lm_head_idx][0], out_tensor, 0, src_offset, bytes_size)
        run_model(self.model_info_p, self.lm_head_idx, 0)

        # greedy_head
        if self.greedy_head_idx != -1 and self.generation_mode == 'greedy':
            self.d2d_bytes_offset(self.input_tensors[self.greedy_head_idx][0], token_tensor, 0, 0, token_tensor.contents.size)
            run_model(self.model_info_p, self.greedy_head_idx, 0)
            token_tensor = self.output_tensors[self.greedy_head_idx][0]
        
        # penalty_sample_head
        # todo
        
        untensor_sync(token_tensor, False, True)
        token = ctypes.cast(token_tensor.contents.data, ctypes.POINTER(ctypes.c_int32)).contents.value
        self.token_length += 1
        free_mask_ptr(attn_mask_ptr)
        return token

    def forward_next(self):
        position_id = [self.token_length - 1]
        c_position_id = (ctypes.c_int * len(position_id))(*position_id)
        position_id_ptr = ctypes.cast(c_position_id, ctypes.c_void_p)

        attn_mask_ptr = get_next_mask_ptr(self.SEQLEN, self.token_length, self.is_bf16)

        # 网络1 embedding_cache
        if self.greedy_head_idx != -1 and self.generation_mode == 'greedy':
            token_tensor = self.output_tensors[self.greedy_head_idx][0]
        # elif self.penalty_sample_head_idx != -1 and self.generation_mode == 'penalty_sample':
        #     token_tensor = self.output_tensors[self.penalty_sample_head_idx][0]
        else:
            token_tensor = self.output_tensors[self.lm_head_idx][0]
        self.d2d_bytes_offset(self.input_tensors[self.embedding_cache_idx][0], token_tensor, 0, 0, token_tensor.contents.size)
        run_model(self.model_info_p, self.embedding_cache_idx, 0)
        out_tensor = self.output_tensors[self.embedding_cache_idx][0]

        # 网络2 attention block_cache
        block_cache_0 = self.block_cache_ids[0]
        bytes_size = self.output_tensors[block_cache_0][1].contents.size
        dst_offset = bytes_size * (self.token_length - 1)
        for i in range(self.num_layers):
            net_id = self.block_cache_ids[i]
            self.d2d_bytes_offset(self.input_tensors[net_id][0], out_tensor, 0, 0, out_tensor.contents.size)
            if i == 0:
                self.s2d_bytes(self.input_tensors[net_id][1], position_id_ptr, 4)
                self.s2d_bytes(self.input_tensors[net_id][2], attn_mask_ptr, 2 * (self.SEQLEN + 1))
            else:
                self.d2d_bytes_offset(self.input_tensors[net_id][1], self.input_tensors[block_cache_0][1], 0, 0, 4)
                self.d2d_bytes_offset(self.input_tensors[net_id][2], self.input_tensors[block_cache_0][2], 0, 0, 2 * (self.SEQLEN + 1))
            
            run_model(self.model_info_p, net_id, 0)
            out_tensor = self.output_tensors[net_id][0]
            self.d2d_bytes_offset(self.input_tensors[net_id][3], self.output_tensors[net_id][1], dst_offset, 0, bytes_size)
            self.d2d_bytes_offset(self.input_tensors[net_id][4], self.output_tensors[net_id][2], dst_offset, 0, bytes_size)
        
        # lm_head
        token_tensor = self.output_tensors[self.lm_head_idx][0]
        self.d2d_bytes_offset(self.input_tensors[self.lm_head_idx][0], out_tensor, 0, 0, out_tensor.contents.size)
        run_model(self.model_info_p, self.lm_head_idx, 0)

        # greedy_head
        if self.greedy_head_idx != -1 and self.generation_mode == 'greedy':
            self.d2d_bytes_offset(self.input_tensors[self.greedy_head_idx][0], token_tensor, 0, 0, token_tensor.contents.size)
            run_model(self.model_info_p, self.greedy_head_idx, 0)
            token_tensor = self.output_tensors[self.greedy_head_idx][0]
        
        # penalty_sample_head
        # todo
        
        untensor_sync(token_tensor, False, True)
        token = ctypes.cast(token_tensor.contents.data, ctypes.POINTER(ctypes.c_int32)).contents.value
        self.token_length += 1
        free_mask_ptr(attn_mask_ptr)
        return token

    def _close(self):
        free_model(self.model_info_p)
        for i in self.input_tensors:
            for j in i:
                untensor_destroy(j)
        for i in self.output_tensors:
            for j in i:
                untensor_destroy(j)

    def __del__(self):
        self._close()


if __name__ == "__main__":
    import time
    start = time.time()
    llm = LLMBaseModel("/data/qwen2.5-3b_int4_seq4096_1dev.bmodel")
    print("load time:", time.time() - start)
    input_ids = [1, 73441, 1836, 5, 23523, 73440, 5, 73441, 43686, 5]

    start = time.time()
    token = llm.forward_first(input_ids)
    print(token)
    print("forward_first time:", time.time() - start)

    for i in range(10):
        start = time.time()
        token = llm.forward_next()
        print(token)
        print("forward_next time:", time.time() - start)