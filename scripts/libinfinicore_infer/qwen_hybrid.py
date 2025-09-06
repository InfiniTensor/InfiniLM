from .base import BaseModel, DataType, DeviceType, KVCacheCStruct, register_model
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
        ("has_qkv_bias", c_char),
    ]


class ModelWeightsCStruct(Structure):
    pass


class QwenHybridModelCStruct(Structure):
    pass


@register_model
class QwenHybridModel(BaseModel):
    @classmethod
    def register_lib(cls, lib):
        """Register QwenHybrid model functions with the library"""
        lib.createQwenHybridModel.restype = POINTER(QwenHybridModelCStruct)
        lib.createQwenHybridModel.argtypes = [
            POINTER(QwenHybridMetaCStruct),
            DeviceType,
            c_int,
            POINTER(c_int),
        ]

        lib.getQwenHybridModelWeights.restype = POINTER(ModelWeightsCStruct)
        lib.getQwenHybridModelWeights.argtypes = [
            POINTER(QwenHybridModelCStruct),
        ]

        lib.destroyQwenHybridModel.argtypes = [POINTER(QwenHybridModelCStruct)]

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

        lib.inferBatchQwenHybrid.argtypes = [
            POINTER(QwenHybridModelCStruct),
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

        lib.forwardBatchQwenHybrid.argtypes = [
            POINTER(QwenHybridModelCStruct),
            POINTER(c_uint),
            c_uint,
            POINTER(c_uint),
            c_uint,
            POINTER(c_uint),
            POINTER(POINTER(KVCacheCStruct)),
            c_void_p,
        ]

        lib.loadModelWeight.argtypes = [
            POINTER(ModelWeightsCStruct),
            c_char_p,
            c_void_p,
        ]

    def create_model(self, meta, device_type, ndev, dev_ids):
        return self.lib.createQwenHybridModel(meta, device_type, ndev, dev_ids)

    def get_model_weights(self, model):
        return self.lib.getQwenHybridModelWeights(model)

    def destroy_model(self, model):
        self.lib.destroyQwenHybridModel(model)

    def create_kv_cache(
        self, nlayer, max_len, nkvh, dk, dv, dtype, device, dev_ids, ndev
    ):
        return self.lib.createKVCache(
            nlayer, max_len, nkvh, dk, dv, dtype, device, dev_ids, ndev
        )

    def drop_kv_cache(self, kv_cache):
        self.lib.dropKVCache(kv_cache)

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
