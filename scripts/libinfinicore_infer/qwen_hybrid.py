from .base import (
    BaseModel,
    DataType,
    DeviceType,
    register_model,
    ModelWeightsCStruct,
)
from ctypes import (
    c_size_t,
    c_uint,
    c_int,
    c_float,
    c_void_p,
    POINTER,
    Structure,
    c_char,
    c_char_p,
)


class QwenHybridMetaCStruct(Structure):
    _fields_ = [
        ("dt_logits", DataType),
        ("dt_linear_w", DataType),
        ("dt_norm_w", DataType),
        ("nlayer", c_size_t),
        ("d", c_size_t),
        ("nh", c_size_t),
        ("nkvh", c_size_t),
        ("dh", c_size_t),
        ("di", c_size_t),
        ("dctx", c_size_t),
        ("dvoc", c_size_t),
        ("epsilon", c_float),
        ("theta", c_float),
        ("end_token", c_uint),
        ("nbit", c_size_t),
        ("quant_group_size", c_size_t),
        ("has_qkv_bias", c_char),
    ]


class QwenHybridModelCStruct(Structure):
    pass


class QwenHybridCacheCStruct(Structure):
    pass


@register_model
class QwenHybridModel(BaseModel):
    @classmethod
    def register_lib(cls, lib):
        """Register QwenHybrid model functions with the library"""
        lib.createQwenHybridWeights.restype = POINTER(ModelWeightsCStruct)
        lib.createQwenHybridWeights.argtypes = [
            POINTER(QwenHybridMetaCStruct),
            DeviceType,
            c_int,
            POINTER(c_int),
        ]

        lib.createQwenHybridModel.restype = POINTER(QwenHybridModelCStruct)
        lib.createQwenHybridModel.argtypes = [
            POINTER(QwenHybridMetaCStruct),
            POINTER(ModelWeightsCStruct),
        ]

        lib.destroyQwenHybridModel.argtypes = [POINTER(QwenHybridModelCStruct)]

        lib.createQwenHybridCache.argtypes = [
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
        lib.createQwenHybridCache.restype = POINTER(QwenHybridCacheCStruct)

        lib.dropQwenHybridCache.argtypes = [POINTER(QwenHybridCacheCStruct)]

        lib.inferBatchQwenHybrid.argtypes = [
            POINTER(QwenHybridModelCStruct),
            POINTER(c_uint),
            c_uint,
            POINTER(c_uint),
            c_uint,
            POINTER(c_uint),
            POINTER(POINTER(QwenHybridCacheCStruct)),
            POINTER(c_float),
            POINTER(c_uint),
            POINTER(c_float),
            POINTER(c_uint),
        ]

        lib.forwardBatchQwenHybrid.argtypes = [
            POINTER(QwenHybridModelCStruct),
            POINTER(c_uint),
            c_uint,
            POINTER(c_uint),
            c_uint,
            POINTER(c_uint),
            POINTER(POINTER(QwenHybridCacheCStruct)),
            c_void_p,
        ]

        lib.loadModelWeight.argtypes = [
            POINTER(ModelWeightsCStruct),
            c_char_p,
            c_void_p,
        ]

    def create_weights(self, meta, device_type, ndev, dev_ids):
        return self.lib.createQwenHybridWeights(meta, device_type, ndev, dev_ids)

    def create_model(self, meta, weights):
        return self.lib.createQwenHybridModel(meta, weights)

    def destroy_model(self, model):
        self.lib.destroyQwenHybridModel(model)

    def create_qwen_hybrid_cache(
        self, nlayer, max_len, nkvh, dk, dv, dtype, device, dev_ids, ndev
    ):
        return self.lib.createQwenHybridCache(
            nlayer, max_len, nkvh, dk, dv, dtype, device, dev_ids, ndev
        )

    def drop_qwen_hybrid_cache(self, qwen_hybrid_cache):
        self.lib.dropQwenHybridCache(qwen_hybrid_cache)

    def load_weight(self, weights, name, data):
        self.lib.loadModelWeight(weights, name.encode("utf-8"), data)

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
        self.lib.inferBatchQwenHybrid(
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
        self.lib.forwardBatchQwenHybrid(
            model, tokens, ntok, req_lens, nreq, req_pos, kv_caches, logits
        )
