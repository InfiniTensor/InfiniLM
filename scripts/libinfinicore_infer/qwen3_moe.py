from .base import BaseModel, DataType, DeviceType, KVCacheCStruct, register_model
from ctypes import (
    c_size_t,
    c_uint,
    c_int,
    c_float,
    c_void_p,
    POINTER,
    Structure,
    CFUNCTYPE,
)


class Qwen3MoEAttentionMetaCStruct(Structure):
    _fields_ = [
        ("dtype", DataType),
        ("hidden_size", c_size_t),
        ("num_heads", c_size_t),
        ("num_kv_head", c_size_t),
        ("head_dim", c_size_t),
        ("rope_theta", c_float),
        ("max_seq_len", c_size_t),
        ("rms_norm_eps", c_float),
    ]


class Qwen3MoEWeightsCStruct(Structure):
    pass


class Qwen3MoEAttentionCStruct(Structure):
    pass


class Qwen3CacheCStruct(Structure):
    pass


load_layer_fn = CFUNCTYPE(None, POINTER(Qwen3MoEWeightsCStruct), c_void_p, c_size_t)
load_layer_linear_fn = CFUNCTYPE(
    None, POINTER(Qwen3MoEWeightsCStruct), c_void_p, c_void_p, c_void_p, c_size_t
)


class Qwen3MoEWeightLoaderCStruct(Structure):
    _fields_ = [
        ("load_attn_norm", load_layer_fn),
        ("load_attn_q_proj", load_layer_linear_fn),
        ("load_attn_k_proj", load_layer_linear_fn),
        ("load_attn_v_proj", load_layer_linear_fn),
        ("load_attn_q_norm", load_layer_fn),
        ("load_attn_k_norm", load_layer_fn),
        ("load_attn_o_proj", load_layer_linear_fn),
    ]


@register_model
class Qwen3MoEModel(BaseModel):
    @classmethod
    def register_lib(cls, lib):
        """Register Qwen3MoE model functions with the library"""
        lib.createQwen3MoEWeightLoader.argtypes = []
        lib.createQwen3MoEWeightLoader.restype = POINTER(
            Qwen3MoEWeightLoaderCStruct
        )

        lib.createQwen3MoEWeights.argtypes = [
            POINTER(Qwen3MoEAttentionMetaCStruct),
            DeviceType,
            c_int,
            POINTER(c_int),
        ]
        lib.createQwen3MoEWeights.restype = POINTER(Qwen3MoEWeightsCStruct)

        lib.createQwen3MoEAttention.argtypes = [
            POINTER(Qwen3MoEAttentionMetaCStruct),
            POINTER(Qwen3MoEWeightsCStruct),
        ]
        lib.createQwen3MoEAttention.restype = POINTER(Qwen3MoEAttentionCStruct)

        lib.destroyQwen3MoEAttention.argtypes = [POINTER(Qwen3MoEAttentionCStruct)]

        lib.createQwen3Cache.argtypes = [
            POINTER(Qwen3MoEAttentionMetaCStruct),
            c_size_t,
            c_size_t,
        ]
        lib.createQwen3Cache.restype = POINTER(Qwen3CacheCStruct)

        lib.forwardQwen3MoEAttention.argtypes = [
            POINTER(Qwen3MoEAttentionCStruct),
            POINTER(Qwen3CacheCStruct),
            c_void_p,
            c_void_p,
        ]

    def create_weight_loader(self):
        return self.lib.createQwen3MoEWeightLoader()

    def create_weights(self, meta, device_type, ndev, dev_ids):
        return self.lib.createQwen3MoEWeights(meta, device_type, ndev, dev_ids)

    def create_model(self, meta, weights):
        return self.lib.createQwen3MoEAttention(meta, weights)

    def destroy_model(self, model):
        self.lib.destroyQwen3MoEAttention(model)

    def create_cache(self, meta, batch_size, seq_len):
        return self.lib.createQwen3Cache(meta, batch_size, seq_len)

    def forward_attention(self, model, kv_cache, input_tensor, output_tensor):
        self.lib.forwardQwen3MoEAttention(model, kv_cache, input_tensor, output_tensor)


