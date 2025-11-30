from .base import BaseModel, DataType, DeviceType, KVCacheCStruct, register_model
from ctypes import c_size_t, c_uint, c_int, c_float, c_void_p, POINTER, Structure, byref


class LLaDAMetaCStruct(Structure): # from config file
    _fields_ = [ # (name, size)
        ("dt_logits", DataType), # c_int
        ("nlayer", c_size_t),
        ("d", c_size_t),
        ("nh", c_size_t),
        ("nkvh", c_size_t),
        ("dh", c_size_t),
        ("di_dense", c_size_t),
        ("di_expert", c_size_t),
        ("dctx", c_size_t),
        ("dvoc", c_size_t),
        ("epsilon", c_float),
        ("theta", c_float),
        ("end_token", c_uint),
        ("num_experts", c_int)
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
        ("ffn_gate_up", POINTER(c_void_p)),
        ("ffn_down", POINTER(c_void_p)),
    ]

class LLaDAModelCStruct(Structure):
    pass


@register_model
class LLaDAModel(BaseModel):
    @classmethod
    def register_lib(cls, lib):
        pass
    def create_model(self, meta, weights, device_type, ndev, dev_ids):
        pass
    def destroy_model(self, model):
        pass

    def create_kv_cache(
        self, nlayer, max_len, nkvh, dk, dv, dtype, device, dev_ids, ndev
    ):
        pass

    def drop_kv_cache(self, kv_cache):
        pass

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
        pass

    def forward_batch(
        self, model, tokens, ntok, req_lens, nreq, req_pos, kv_caches, logits
    ):
        pass
