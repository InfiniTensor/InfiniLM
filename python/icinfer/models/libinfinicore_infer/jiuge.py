from .base import BaseModel, DataType, DeviceType, KVCacheCStruct, register_model
from ctypes import c_size_t, c_uint, c_int, c_float, c_void_p, POINTER, Structure, byref, c_bool


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

        lib.createPagedKVCache.argtypes = [
            c_size_t,
            c_size_t,
            c_size_t,
            c_size_t,
            DataType,
            DeviceType,
            POINTER(c_int),
            c_size_t,
            c_size_t,
            c_size_t,
        ]
        lib.createPagedKVCache.restype = POINTER(KVCacheCStruct)

        lib.dropKVCache.argtypes = [POINTER(KVCacheCStruct)]

        # lib.inferBatchJiuge.argtypes = [
        #     POINTER(JiugeModelCStruct),
        #     POINTER(c_uint),
        #     c_uint,
        #     POINTER(c_uint),
        #     c_uint,
        #     POINTER(c_uint),
        #     POINTER(POINTER(KVCacheCStruct)),
        #     POINTER(c_float),
        #     POINTER(c_uint),
        #     POINTER(c_float),
        #     POINTER(c_uint),
        # ]

        # lib.forwardBatchJiuge.argtypes = [
        #     POINTER(JiugeModelCStruct),
        #     POINTER(c_uint),
        #     c_uint,
        #     POINTER(c_uint),
        #     c_uint,
        #     POINTER(c_uint),
        #     POINTER(POINTER(KVCacheCStruct)),
        #     c_void_p,
        # ]

        lib.inferBatchJiuge.argtypes = [
            POINTER(JiugeModelCStruct),  # struct JiugeModel const *
            POINTER(c_uint),  # unsigned int const *tokens
            c_uint,  # unsigned int ntok
            POINTER(c_uint),  # unsigned int const *req_lens
            c_uint,  # unsigned int nreq
            POINTER(c_uint),  # unsigned int const *req_pos
            POINTER(POINTER(KVCacheCStruct)),  # struct KVCache **kv_caches
            POINTER(c_int),  # unsigned int const *block_tables
            POINTER(c_int),  # unsigned int const *slot_mapping
            POINTER(c_float),  # float temperature
            POINTER(c_uint),  # unsigned int topk
            POINTER(c_float),  # float topp
            c_uint,  # unsigned int is_prefill
            c_bool,  # bool enable_paged_attn
            POINTER(c_uint),  # unsigned int *output
        ]
        lib.forwardBatchJiuge.argtypes = [
            POINTER(JiugeModelCStruct),  # struct JiugeModel const *
            POINTER(c_uint),  # unsigned int const *tokens
            c_uint,  # unsigned int ntok
            POINTER(c_uint),  # unsigned int const *req_lens
            c_uint,  # unsigned int nreq
            POINTER(c_uint),  # unsigned int const *req_pos
            POINTER(POINTER(KVCacheCStruct)),  # struct KVCache **kv_caches
            POINTER(c_int),  # unsigned int const *block_tables
            POINTER(c_int),  # unsigned int const *slot_mapping
            c_uint,  # unsigned int is_prefill
            c_bool,  # bool enable_paged_attn
            c_void_p,  # void *logits
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

    def create_paged_kv_cache(
        self, nlayer, max_len, nkvh, dk, dv, dtype, device, dev_ids, ndev, max_kvcache_tokens
    ):
        return self.lib.createPagedKVCache(
            nlayer, max_len, nkvh, dk, dv, dtype, device, dev_ids, ndev, max_kvcache_tokens
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
        block_tables,
        slot_mapping,
        temperature,
        topk,
        topp,
        is_prefill,
        enable_paged_attn,
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
            block_tables,
            slot_mapping,
            temperature,
            topk,
            topp,
            is_prefill,
            enable_paged_attn,
            output,
        )

    def forward_batch(
        self, model, tokens, ntok, req_lens, nreq, req_pos, 
        kv_caches, block_tables, slot_mapping, is_prefill, enable_paged_attn, logits
    ):
        self.lib.forwardBatchJiuge(
            model, tokens, ntok, req_lens, nreq, req_pos, kv_caches, block_tables, slot_mapping, is_prefill, enable_paged_attn, logits
        )
