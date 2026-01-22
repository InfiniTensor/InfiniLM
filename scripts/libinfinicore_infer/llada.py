from .base import BaseModel, DataType, DeviceType, KVCacheCStruct, register_model
from ctypes import c_size_t, c_uint, c_int, c_float, c_void_p, POINTER, Structure, byref


class LLaDAMetaCStruct(Structure): # from config file
    _fields_ = [ # (name, size)
        ("dt_logits", DataType),      # 4 bytes
        ("_pad0", c_uint),            # 填充 4 bytes，使下一个 c_size_t 对齐
        ("nlayer", c_size_t),         # 8 bytes
        ("d", c_size_t),
        ("nh", c_size_t),
        ("nkvh", c_size_t),
        ("dh", c_size_t),
        ("di_dense", c_size_t),
        ("di_expert", c_size_t),
        ("dctx", c_size_t),
        ("dvoc", c_size_t),
        ("epsilon", c_float),         # 4 bytes
        ("theta", c_float),
        ("end_token", c_uint),        # 4 bytes
        ("_pad1", c_uint),            # 填充 4 bytes，使下一个 size_t 对齐
        ("num_experts", c_size_t),
    ] # equal to c structure in c


class LLaDAWeightsCStruct(Structure):
     _fields_ = [
        ("nlayer", c_size_t),
        ("dt_norm", DataType),
        ("dt_mat", DataType),
        ("transpose_linear_weights", c_int),
        ("input_embd", c_void_p),
        ("output_norm", c_void_p),
        ("output_embd", c_void_p),
        ("attn_norm", POINTER(c_void_p)),
        ("attn_qkv", POINTER(c_void_p)),
        ("attn_qkv_b", POINTER(c_void_p)),
        ("attn_q_norm", POINTER(c_void_p)),
        ("attn_k_norm", POINTER(c_void_p)),
        ("attn_o", POINTER(c_void_p)),
        ("ffn_norm", POINTER(c_void_p)),

        ("expert_gate", POINTER(c_void_p)),
        ("expert_up", POINTER(c_void_p)),
        ("expert_down", POINTER(c_void_p)),
        ("router", POINTER(c_void_p)),
    ]

class LLaDAModelCStruct(Structure):
    pass


@register_model
class LLaDAModel(BaseModel):
    @classmethod
    def register_lib(cls, lib):
        # 此处实现参数列表的对应
        # TODO:
            # createJiugeModel  C++ interface
            # destoryJIugeModel C++ interface 
            # inferBatchJiuge   C++ interface
            # forwardBatchJiuge C++ interface 
        # TODO: 根据最后4个的实现完善参数列表
        lib.createLLaDAModel.restype = POINTER(LLaDAModelCStruct) # OK
        lib.createLLaDAModel.argtypes = [
            POINTER(LLaDAMetaCStruct),
            POINTER(LLaDAWeightsCStruct),
            DeviceType,
            c_int,
            POINTER(c_int), # const --> Pointer
        ] # OK

        # lib.destroyJiugeModel.argtypes = [POINTER(LLaDAModelCStruct)]

        lib.createKVCache.argtypes = [
            c_size_t,
            c_size_t,
            c_size_t,
            c_size_t,
            c_size_t,
            DataType,
            DeviceType,
            POINTER(c_int),
            c_size_t,
        ]
        lib.createKVCache.restype = POINTER(KVCacheCStruct)

        lib.dropKVCache.argtypes = [POINTER(KVCacheCStruct)]

        lib.inferBatchLLaDA.argtypes = [
            POINTER(LLaDAModelCStruct),
            POINTER(c_uint),
            c_uint,
            POINTER(c_uint),
            c_uint,
            POINTER(c_uint),
            POINTER(POINTER(KVCacheCStruct)),
            POINTER(c_float),
            POINTER(c_uint),
            POINTER(c_float),
            POINTER(c_uint),
        ]

        lib.forwardBatchLLaDA.argtypes = [
            POINTER(LLaDAModelCStruct),
            POINTER(c_uint),
            c_uint,
            POINTER(c_uint),
            c_uint,
            POINTER(c_uint),
            POINTER(POINTER(KVCacheCStruct)),
            c_void_p,
        ]
    def create_model(self, meta, weights, device_type, ndev, dev_ids):
        # TODO:
        return self.lib.createLLaDAModel(meta, weights, device_type, ndev, dev_ids)
    
    def destroy_model(self, model):
        pass

    def create_kv_cache(
        self, nlayer, max_len, nkvh, dk, dv, dtype, device, dev_ids, ndev
    ):
        return self.lib.createKVCache(
            nlayer, max_len, nkvh, dk, dv, dtype, device, dev_ids, ndev
        )

    def drop_kv_cache(self, kv_cache):
        self.lib.dropKVCache(kv_cache)

    def infer_batch(
        self,
        model,
        tokens,
        ntok,
        req_lens,
        nreq,
        req_pos,
        kv_caches,
        temperature,
        topk,
        topp,
        output,
    ):
        self.lib.inferBatchLLaDA(
            model,
            tokens,
            ntok,
            req_lens,
            nreq,
            req_pos,
            kv_caches,
            temperature,
            topk,
            topp,
            output,
        )

    def forward_batch(
        self, model, tokens, ntok, req_lens, nreq, req_pos, kv_caches, logits
    ):
        self.lib.forwardBatchLLaDA(
            model, tokens, ntok, req_lens, nreq, req_pos, kv_caches, logits
        )
