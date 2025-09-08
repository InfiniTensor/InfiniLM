from .base import (
    BaseModel,
    DataType,
    DeviceType,
    register_model,
    ModelWeightsCStruct,
    KVCacheCStruct,
    MambaCacheCStruct,
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
        # common
        ("dtype", DataType),
        ("nlayer", c_size_t),
        ("d", c_size_t),
        ("dctx", c_size_t),
        ("dvoc", c_size_t),
        ("epsilon", c_float),
        ("end_token", c_uint),
        # mha
        ("nh", c_size_t),
        ("nkvh", c_size_t),
        ("dh", c_size_t),
        ("theta", c_float),
        # linear attention
        ("l_conv_dim", c_size_t),
        ("l_expand", c_size_t),
        ("l_n_k_head", c_size_t),
        ("l_k_dim", c_size_t),
        ("l_n_v_head", c_size_t),
        ("l_v_dim", c_size_t),
        # moe
        ("nexperts", c_size_t),
        ("kexperts", c_size_t),
        ("shared_di", c_size_t),
        ("moe_di", c_size_t),
    ]


class QwenHybridModelCStruct(Structure):
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

        lib.createMambaCache.argtypes = [
            c_size_t,
            DataType,
            DeviceType,
            POINTER(c_int),
            c_size_t,
        ]
        lib.createMambaCache.restype = POINTER(MambaCacheCStruct)

        lib.dropMambaCache.argtypes = [POINTER(MambaCacheCStruct)]

        lib.inferBatchQwenHybrid.argtypes = [
            POINTER(QwenHybridModelCStruct),
            POINTER(c_uint),
            c_uint,
            POINTER(c_uint),
            c_uint,
            POINTER(c_uint),
            POINTER(POINTER(KVCacheCStruct)),
            POINTER(POINTER(MambaCacheCStruct)),
            POINTER(c_float),
            POINTER(c_uint),
            POINTER(c_float),
            POINTER(c_uint),
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

    def create_kv_cache(
        self, nlayer, max_len, nkvh, dk, dv, dtype, device, dev_ids, ndev
    ):
        return self.lib.createKVCache(
            nlayer, max_len, nkvh, dk, dv, dtype, device, dev_ids, ndev
        )

    def drop_kv_cache(self, kv_cache):
        self.lib.dropKVCache(kv_cache)

    def create_mamba_cache(self, nlayer, dtype, device, dev_ids, ndev):
        return self.lib.createMambaCache(nlayer, dtype, device, dev_ids, ndev)

    def drop_mamba_cache(self, mamba_cache):
        self.lib.dropMambaCache(mamba_cache)

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
