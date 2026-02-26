from .base import BaseModel, DataType, DeviceType, KVCacheCStruct, register_model, ModelWeightsCStruct
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


class Qwen3VLMetaCStruct(Structure):
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
        ("has_qkv_bias", c_char),
        # vision encoder parameters
        ("use_qk_norm", c_char),
        ("vision_hidden_size", c_size_t),
        ("vision_layers", c_size_t),
        ("vision_heads", c_size_t),
        ("patch_size", c_size_t),
        ("img_size", c_size_t),
        # token ids
        ("image_token_id", c_uint),
        ("video_token_id", c_uint),
    ]


class Qwen3VLModelCStruct(Structure):
    pass


@register_model
class Qwen3VLModel(BaseModel):
    @classmethod
    def register_lib(cls, lib):
        """Register Qwen3VL model functions with the library"""
        lib.createQwen3VLWeights.restype = POINTER(ModelWeightsCStruct)
        lib.createQwen3VLWeights.argtypes = [
            POINTER(Qwen3VLMetaCStruct),
            DeviceType,
            c_int,
            POINTER(c_int),
        ]

        lib.createQwen3VLModel.restype = POINTER(Qwen3VLModelCStruct)
        lib.createQwen3VLModel.argtypes = [
            POINTER(Qwen3VLMetaCStruct),
            POINTER(ModelWeightsCStruct),
        ]

        lib.destroyQwen3VLModel.argtypes = [POINTER(Qwen3VLModelCStruct)]

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

        lib.inferBatchQwen3VL.argtypes = [
            POINTER(Qwen3VLModelCStruct),
            POINTER(c_uint),
            c_uint,
            POINTER(c_uint),
            c_uint,
            POINTER(c_uint),
            POINTER(c_uint),
            c_uint,
            POINTER(c_uint),  # llm_pos_ids
            c_uint,          # llm_pos_ids_len
            POINTER(c_uint),  # rope_section
            c_uint,          # rope_section_len
            c_void_p,
            POINTER(POINTER(KVCacheCStruct)),
            POINTER(c_float),
            POINTER(c_uint),
            POINTER(c_float),
            POINTER(c_uint),
        ]

        lib.forwardBatchQwen3VL.argtypes = [
            POINTER(Qwen3VLModelCStruct),
            POINTER(c_uint),
            c_uint,
            POINTER(c_uint),
            c_uint,
            POINTER(c_uint),
            POINTER(c_uint),
            c_uint,
            POINTER(c_uint),  # llm_pos_ids
            c_uint,          # llm_pos_ids_len
            POINTER(c_uint),  # rope_section
            c_uint,          # rope_section_len
            c_void_p,
            POINTER(POINTER(KVCacheCStruct)),
            c_void_p,
        ]

        lib.loadModelWeight.argtypes = [
            POINTER(ModelWeightsCStruct),
            c_char_p,
            c_void_p,
        ]

    def create_weights(self, meta, device_type, ndev, dev_ids):
        return self.lib.createQwen3VLWeights(meta, device_type, ndev, dev_ids)

    def create_model(self, meta, weights):
        return self.lib.createQwen3VLModel(meta, weights)

    def destroy_model(self, model):
        self.lib.destroyQwen3VLModel(model)

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
        pos_ids,
        pos_ids_len,
        llm_pos_ids,
        llm_pos_ids_len,
        rope_section,
        rope_section_len,
        pixel_values,
        kv_caches,
        temperature,
        topk,
        topp,
        output,
    ):
        self.lib.inferBatchQwen3VL(
            model,
            tokens,
            ntok,
            req_lens,
            nreq,
            req_pos,
            pos_ids,
            pos_ids_len,
            llm_pos_ids,
            llm_pos_ids_len,
            rope_section,
            rope_section_len,
            pixel_values,
            kv_caches,
            temperature,
            topk,
            topp,
            output,
        )

    def forward_batch(
        self, model, tokens, ntok, req_lens, nreq, req_pos, pos_ids, pos_ids_len,
        llm_pos_ids, llm_pos_ids_len, rope_section, rope_section_len,
        pixel_values, kv_caches, logits
    ):
        self.lib.forwardBatchQwen3VL(
            model, tokens, ntok, req_lens, nreq, req_pos, pos_ids, pos_ids_len,
            llm_pos_ids, llm_pos_ids_len, rope_section, rope_section_len,
            pixel_values, kv_caches, logits
        )
