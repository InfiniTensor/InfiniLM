from .base import (
    BaseModel,
    DataType,
    DeviceType,
    KVCacheCStruct,
    KVCompressionConfigCStruct,
    register_model,
)
from ctypes import c_size_t, c_uint, c_int, c_float, c_void_p, POINTER, Structure, byref


class JiugeMetaCStruct(Structure):
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
    ]


class JiugeWeightsCStruct(Structure):
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


class JiugeModelCStruct(Structure):
    pass


@register_model
class JiugeModel(BaseModel):
    @classmethod
    def register_lib(cls, lib):
        lib.createJiugeModel.restype = POINTER(JiugeModelCStruct)
        lib.createJiugeModel.argtypes = [
            POINTER(JiugeMetaCStruct),
            POINTER(JiugeWeightsCStruct),
            DeviceType,
            c_int,
            POINTER(c_int),
        ]

        lib.destroyJiugeModel.argtypes = [POINTER(JiugeModelCStruct)]

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

        lib.compressKVCacheInplace.argtypes = [
            POINTER(KVCacheCStruct),
            c_uint,
            POINTER(KVCompressionConfigCStruct),
        ]
        lib.compressKVCacheInplace.restype = c_uint

        lib.inferBatchJiuge.argtypes = [
            POINTER(JiugeModelCStruct),
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

        lib.inferBatchJiugeWithLogits.argtypes = [
            POINTER(JiugeModelCStruct),
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
            c_void_p,  # logits
        ]

        lib.inferBatchJiugeEx.argtypes = [
            POINTER(JiugeModelCStruct),
            POINTER(c_uint),
            c_uint,
            POINTER(c_uint),
            c_uint,
            POINTER(c_uint),  # req_pos
            POINTER(c_uint),  # kv_pos
            POINTER(POINTER(KVCacheCStruct)),
            POINTER(c_float),
            POINTER(c_uint),
            POINTER(c_float),
            POINTER(c_uint),
        ]

        lib.inferBatchJiugeExWithLogits.argtypes = [
            POINTER(JiugeModelCStruct),
            POINTER(c_uint),
            c_uint,
            POINTER(c_uint),
            c_uint,
            POINTER(c_uint),  # req_pos
            POINTER(c_uint),  # kv_pos
            POINTER(POINTER(KVCacheCStruct)),
            POINTER(c_float),
            POINTER(c_uint),
            POINTER(c_float),
            POINTER(c_uint),
            c_void_p,  # logits
        ]

        lib.inferBatchJiugeWithOverrides.argtypes = [
            POINTER(JiugeModelCStruct),
            POINTER(c_uint),
            c_uint,
            POINTER(c_uint),
            c_uint,
            POINTER(c_uint),
            POINTER(POINTER(KVCacheCStruct)),
            c_uint,  # n_override
            POINTER(c_uint),  # override_pos
            c_void_p,  # override_embeds
            POINTER(c_float),
            POINTER(c_uint),
            POINTER(c_float),
            POINTER(c_uint),
        ]

        lib.inferBatchJiugeWithOverridesEx.argtypes = [
            POINTER(JiugeModelCStruct),
            POINTER(c_uint),
            c_uint,
            POINTER(c_uint),
            c_uint,
            POINTER(c_uint),  # req_pos
            POINTER(c_uint),  # kv_pos
            POINTER(POINTER(KVCacheCStruct)),
            c_uint,  # n_override
            POINTER(c_uint),  # override_pos
            c_void_p,  # override_embeds
            POINTER(c_float),
            POINTER(c_uint),
            POINTER(c_float),
            POINTER(c_uint),
        ]

        lib.inferBatchJiugeWithOverridesWithLogits.argtypes = [
            POINTER(JiugeModelCStruct),
            POINTER(c_uint),
            c_uint,
            POINTER(c_uint),
            c_uint,
            POINTER(c_uint),
            POINTER(POINTER(KVCacheCStruct)),
            c_uint,  # n_override
            POINTER(c_uint),  # override_pos
            c_void_p,  # override_embeds
            POINTER(c_float),
            POINTER(c_uint),
            POINTER(c_float),
            POINTER(c_uint),
            c_void_p,  # logits
        ]

        lib.forwardBatchJiuge.argtypes = [
            POINTER(JiugeModelCStruct),
            POINTER(c_uint),
            c_uint,
            POINTER(c_uint),
            c_uint,
            POINTER(c_uint),
            POINTER(POINTER(KVCacheCStruct)),
            c_void_p,
        ]

        lib.forwardBatchJiugeEx.argtypes = [
            POINTER(JiugeModelCStruct),
            POINTER(c_uint),
            c_uint,
            POINTER(c_uint),
            c_uint,
            POINTER(c_uint),  # req_pos
            POINTER(c_uint),  # kv_pos
            POINTER(POINTER(KVCacheCStruct)),
            c_void_p,  # logits
        ]

        lib.forwardBatchJiugeWithOverrides.argtypes = [
            POINTER(JiugeModelCStruct),
            POINTER(c_uint),
            c_uint,
            POINTER(c_uint),
            c_uint,
            POINTER(c_uint),
            POINTER(POINTER(KVCacheCStruct)),
            c_uint,  # n_override
            POINTER(c_uint),  # override_pos
            c_void_p,  # override_embeds
            c_void_p,  # logits
        ]

        lib.forwardBatchJiugeWithOverridesEx.argtypes = [
            POINTER(JiugeModelCStruct),
            POINTER(c_uint),
            c_uint,
            POINTER(c_uint),
            c_uint,
            POINTER(c_uint),  # req_pos
            POINTER(c_uint),  # kv_pos
            POINTER(POINTER(KVCacheCStruct)),
            c_uint,  # n_override
            POINTER(c_uint),  # override_pos
            c_void_p,  # override_embeds
            c_void_p,  # logits
        ]

    def create_model(self, meta, weights, device_type, ndev, dev_ids):
        return self.lib.createJiugeModel(meta, weights, device_type, ndev, dev_ids)

    def destroy_model(self, model):
        self.lib.destroyJiugeModel(model)

    def create_kv_cache(
        self, nlayer, max_len, nkvh, dk, dv, dtype, device, dev_ids, ndev
    ):
        return self.lib.createKVCache(
            nlayer, max_len, nkvh, dk, dv, dtype, device, dev_ids, ndev
        )

    def drop_kv_cache(self, kv_cache):
        self.lib.dropKVCache(kv_cache)

    def compress_kv_cache_inplace(self, kv_cache, seq_len, cfg: KVCompressionConfigCStruct):
        return self.lib.compressKVCacheInplace(kv_cache, seq_len, byref(cfg))

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
        self.lib.inferBatchJiuge(
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

    def infer_batch_with_logits(
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
        logits,
    ):
        self.lib.inferBatchJiugeWithLogits(
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
            logits,
        )

    def infer_batch_ex(
        self,
        model,
        tokens,
        ntok,
        req_lens,
        nreq,
        req_pos,
        kv_pos,
        kv_caches,
        temperature,
        topk,
        topp,
        output,
    ):
        self.lib.inferBatchJiugeEx(
            model,
            tokens,
            ntok,
            req_lens,
            nreq,
            req_pos,
            kv_pos,
            kv_caches,
            temperature,
            topk,
            topp,
            output,
        )

    def infer_batch_ex_with_logits(
        self,
        model,
        tokens,
        ntok,
        req_lens,
        nreq,
        req_pos,
        kv_pos,
        kv_caches,
        temperature,
        topk,
        topp,
        output,
        logits,
    ):
        self.lib.inferBatchJiugeExWithLogits(
            model,
            tokens,
            ntok,
            req_lens,
            nreq,
            req_pos,
            kv_pos,
            kv_caches,
            temperature,
            topk,
            topp,
            output,
            logits,
        )

    def infer_batch_with_overrides(
        self,
        model,
        tokens,
        ntok,
        req_lens,
        nreq,
        req_pos,
        kv_caches,
        n_override,
        override_pos,
        override_embeds,
        temperature,
        topk,
        topp,
        output,
    ):
        self.lib.inferBatchJiugeWithOverrides(
            model,
            tokens,
            ntok,
            req_lens,
            nreq,
            req_pos,
            kv_caches,
            n_override,
            override_pos,
            override_embeds,
            temperature,
            topk,
            topp,
            output,
        )

    def infer_batch_with_overrides_ex(
        self,
        model,
        tokens,
        ntok,
        req_lens,
        nreq,
        req_pos,
        kv_pos,
        kv_caches,
        n_override,
        override_pos,
        override_embeds,
        temperature,
        topk,
        topp,
        output,
    ):
        self.lib.inferBatchJiugeWithOverridesEx(
            model,
            tokens,
            ntok,
            req_lens,
            nreq,
            req_pos,
            kv_pos,
            kv_caches,
            n_override,
            override_pos,
            override_embeds,
            temperature,
            topk,
            topp,
            output,
        )

    def infer_batch_with_overrides_with_logits(
        self,
        model,
        tokens,
        ntok,
        req_lens,
        nreq,
        req_pos,
        kv_caches,
        n_override,
        override_pos,
        override_embeds,
        temperature,
        topk,
        topp,
        output,
        logits,
    ):
        self.lib.inferBatchJiugeWithOverridesWithLogits(
            model,
            tokens,
            ntok,
            req_lens,
            nreq,
            req_pos,
            kv_caches,
            n_override,
            override_pos,
            override_embeds,
            temperature,
            topk,
            topp,
            output,
            logits,
        )

    def forward_batch(
        self, model, tokens, ntok, req_lens, nreq, req_pos, kv_caches, logits
    ):
        self.lib.forwardBatchJiuge(
            model, tokens, ntok, req_lens, nreq, req_pos, kv_caches, logits
        )

    def forward_batch_ex(
        self,
        model,
        tokens,
        ntok,
        req_lens,
        nreq,
        req_pos,
        kv_pos,
        kv_caches,
        logits,
    ):
        self.lib.forwardBatchJiugeEx(
            model,
            tokens,
            ntok,
            req_lens,
            nreq,
            req_pos,
            kv_pos,
            kv_caches,
            logits,
        )

    def forward_batch_with_overrides(
        self,
        model,
        tokens,
        ntok,
        req_lens,
        nreq,
        req_pos,
        kv_caches,
        n_override,
        override_pos,
        override_embeds,
        logits,
    ):
        self.lib.forwardBatchJiugeWithOverrides(
            model,
            tokens,
            ntok,
            req_lens,
            nreq,
            req_pos,
            kv_caches,
            n_override,
            override_pos,
            override_embeds,
            logits,
        )

    def forward_batch_with_overrides_ex(
        self,
        model,
        tokens,
        ntok,
        req_lens,
        nreq,
        req_pos,
        kv_pos,
        kv_caches,
        n_override,
        override_pos,
        override_embeds,
        logits,
    ):
        self.lib.forwardBatchJiugeWithOverridesEx(
            model,
            tokens,
            ntok,
            req_lens,
            nreq,
            req_pos,
            kv_pos,
            kv_caches,
            n_override,
            override_pos,
            override_embeds,
            logits,
        )
