from .base import BaseModel, DataType, DeviceType, KVCacheCStruct
from .model_register import ModelRegister
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


class DeepSeekV3MetaCStruct(Structure):
    _fields_ = [
        ("dt_logits", DataType),
        ("dt_norm", DataType),
        ("dt_quant_weight", DataType),
        ("dt_quant_scale", DataType),
        ("dt_quant_zero", DataType),
        ("dt_gate_weight", DataType),
        ("dt_gate_bias", DataType),
        ("n_sparse_layer", c_size_t),
        ("n_dense_layer", c_size_t),
        ("d", c_size_t),
        ("nh", c_size_t),
        ("nkvh", c_size_t),
        ("d_rope", c_size_t),
        ("d_nope", c_size_t),
        ("r_q", c_size_t),
        ("r_kv", c_size_t),
        ("d_qk", c_size_t),
        ("d_v", c_size_t),
        ("routed_scale", c_float),
        ("nexperts", c_size_t),
        ("kexperts", c_size_t),
        ("di", c_size_t),
        ("di_moe", c_size_t),
        ("dctx", c_size_t),
        ("dvoc", c_size_t),
        ("epsilon", c_float),
        ("rope_theta", c_float),
        ("end_token", c_uint),
    ]


class DeepSeekV3WeightsCStruct(Structure):
    pass


class DeepSeekV3ModelCStruct(Structure):
    pass


class DeepSeekV3CacheCStruct(Structure):
    pass


load_global_fn = CFUNCTYPE(None, POINTER(DeepSeekV3WeightsCStruct), c_void_p)
load_layer_fn = CFUNCTYPE(None, POINTER(DeepSeekV3WeightsCStruct), c_void_p, c_size_t)
load_layer_linear_fn = CFUNCTYPE(
    None, POINTER(DeepSeekV3WeightsCStruct), c_void_p, c_void_p, c_void_p, c_size_t
)
load_layer_mlp_fn = CFUNCTYPE(
    None,
    POINTER(DeepSeekV3WeightsCStruct),
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
    c_size_t,
)
load_layer_expert_mlp_fn = CFUNCTYPE(
    None,
    POINTER(DeepSeekV3WeightsCStruct),
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
    c_size_t,
    c_size_t,
)


class DeepSeekV3WeightLoaderCStruct(Structure):
    _fields_ = [
        ("load_input_embd", load_global_fn),
        ("load_output_norm", load_global_fn),
        ("load_output_embd", load_global_fn),
        ("load_attn_norm", load_layer_fn),
        ("load_attn_q_a_proj", load_layer_linear_fn),
        ("load_attn_q_a_layernorm", load_layer_fn),
        ("load_attn_q_b_proj", load_layer_linear_fn),
        ("load_attn_kv_a_proj_with_mqa", load_layer_linear_fn),
        ("load_attn_kv_a_layernorm", load_layer_fn),
        ("load_attn_kv_b_proj", load_layer_linear_fn),
        ("load_attn_o_proj", load_layer_linear_fn),
        ("load_mlp_norm", load_layer_fn),
        ("load_mlp_dense", load_layer_mlp_fn),
        ("load_mlp_gate_weight", load_layer_fn),
        ("load_mlp_gate_bias", load_layer_fn),
        ("load_mlp_shared_experts", load_layer_mlp_fn),
        ("load_mlp_experts", load_layer_expert_mlp_fn),
    ]


@ModelRegister.model
class DeepSeekV3Model(BaseModel):
    def __init__(self):
        super().__init__()

    @classmethod
    def register_lib(cls, lib):
        """Register DeepSeekV3 model functions with the library"""
        lib.createDeepSeekV3WeightLoader.argtypes = []
        lib.createDeepSeekV3WeightLoader.restype = POINTER(
            DeepSeekV3WeightLoaderCStruct
        )

        lib.createDeepSeekV3Weights.argtypes = [
            POINTER(DeepSeekV3MetaCStruct),
            DeviceType,
            c_int,
            POINTER(c_int),
        ]
        lib.createDeepSeekV3Weights.restype = POINTER(DeepSeekV3WeightsCStruct)

        lib.createDeepSeekV3Model.argtypes = [
            POINTER(DeepSeekV3MetaCStruct),
            POINTER(DeepSeekV3WeightsCStruct),
        ]
        lib.createDeepSeekV3Model.restype = POINTER(DeepSeekV3ModelCStruct)

        lib.destroyDeepSeekV3Model.argtypes = [POINTER(DeepSeekV3ModelCStruct)]

        lib.createDeepSeekV3Cache.argtypes = [POINTER(DeepSeekV3ModelCStruct)]
        lib.createDeepSeekV3Cache.restype = POINTER(DeepSeekV3CacheCStruct)

        lib.dropDeepSeekV3Cache.argtypes = [
            POINTER(DeepSeekV3ModelCStruct),
            POINTER(DeepSeekV3CacheCStruct),
        ]

        lib.inferBatchDeepSeekV3.argtypes = [
            POINTER(DeepSeekV3ModelCStruct),
            POINTER(c_uint),
            c_uint,
            POINTER(c_uint),
            c_uint,
            POINTER(c_uint),
            POINTER(POINTER(DeepSeekV3CacheCStruct)),
            POINTER(c_float),
            POINTER(c_uint),
            POINTER(c_float),
            POINTER(c_uint),
        ]

    def create_weight_loader(self):
        return self.lib.createDeepSeekV3WeightLoader()

    def create_weights(self, meta, device_type, ndev, dev_ids):
        return self.lib.createDeepSeekV3Weights(meta, device_type, ndev, dev_ids)

    def create_model(self, meta, weights):
        return self.lib.createDeepSeekV3Model(meta, weights)

    def destroy_model(self, model):
        self.lib.destroyDeepSeekV3Model(model)

    def create_cache(self, model):
        return self.lib.createDeepSeekV3Cache(model)

    def drop_cache(self, model, cache):
        self.lib.dropDeepSeekV3Cache(model, cache)

    def infer_batch(
        self,
        model,
        tokens,
        ntok,
        req_lens,
        nreq,
        req_pos,
        caches,
        temperature,
        topk,
        topp,
        output,
    ):
        self.lib.inferBatchDeepSeekV3(
            model,
            tokens,
            ntok,
            req_lens,
            nreq,
            req_pos,
            caches,
            temperature,
            topk,
            topp,
            output,
        )
