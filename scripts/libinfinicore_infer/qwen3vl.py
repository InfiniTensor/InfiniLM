from .base import BaseModel, DataType, DeviceType, KVCacheCStruct, register_model
from ctypes import (
    c_size_t,
    c_uint,
    c_int,
    c_float,
    c_void_p,
    c_bool,
    POINTER,
    Structure,
    CFUNCTYPE,
)


class TextMetaCStruct(Structure):
    _fields_ = [
        ("bos_token_id", c_size_t),
        ("eos_token_id", c_size_t),
        ("head_dim", c_size_t),
        ("hidden_size", c_size_t),
        ("initializer_range", c_float),
        ("intermediate_size", c_size_t),
        ("max_tokens", c_size_t),
        ("num_attention_heads", c_size_t),
        ("num_hidden_layers", c_size_t),
        ("num_key_value_heads", c_size_t),
        ("rms_norm_eps", c_float),
        ("mrope_section", c_size_t * 3),
        ("rope_theta", c_size_t),
        ("vocab_size", c_size_t),
    ]


class VisMetaCStruct(Structure):
    _fields_ = [
        ("depth", c_size_t),
        ("deepstack_visual_indexes", c_size_t * 3),
        ("hidden_size", c_size_t),
        ("in_channels", c_size_t),
        ("initializer_range", c_float),
        ("intermediate_size", c_size_t),
        ("num_heads", c_size_t),
        ("num_position_embeddings", c_size_t),
        ("out_hidden_size", c_size_t),
        ("patch_size", c_size_t),
        ("spatial_merge_size", c_size_t),
        ("temporal_patch_size", c_size_t),
    ]


class Qwen3vlMetaCStruct(Structure):
    _fields_ = [
        ("dtype", DataType),
        ("text_meta", TextMetaCStruct),
        ("vis_meta", VisMetaCStruct),
        # Token ids
        ("image_token_id", c_size_t),
        ("video_token_id", c_size_t),
        ("vision_end_token_id", c_size_t),
        ("vision_start_token_id", c_size_t),
    ]


class Qwen3vlWeightsCStruct(Structure):
    pass


class Qwen3vlModelCStruct(Structure):
    pass


class Qwen3vlCacheCStruct(Structure):
    pass


load_global_fn = CFUNCTYPE(None, POINTER(Qwen3vlWeightsCStruct), c_void_p)
load_layer_fn = CFUNCTYPE(None, POINTER(Qwen3vlWeightsCStruct), c_void_p, c_size_t)


class Qwen3vlLangWeightLoaderCStruct(Structure):
    _fields_ = [
        # Global
        ("load_input_embd", load_global_fn),
        ("load_output_norm", load_global_fn),
        ("load_output_embd", load_global_fn),
        # Attention
        ("load_attn_norm", load_layer_fn),
        ("load_attn_q_norm", load_layer_fn),
        ("load_attn_k_norm", load_layer_fn),
        ("load_attn_qkv_proj", load_layer_fn),
        ("load_attn_o_proj", load_layer_fn),
        # MLP
        ("load_mlp_norm", load_layer_fn),
        ("load_mlp_gate_up", load_layer_fn),
        ("load_mlp_down", load_layer_fn),
    ]


class Qwen3vlVisWeightLoaderCStruct(Structure):
    _fields_ = [
        # Patch embed
        ("load_patch_embed_weight", load_global_fn),
        ("load_patch_embed_bias", load_global_fn),
        ("load_pos_embed_weight", load_global_fn),
        # Blocks attention
        ("load_attn_proj_weight", load_layer_fn),
        ("load_attn_proj_bias", load_layer_fn),
        ("load_attn_qkv_weight", load_layer_fn),
        ("load_attn_qkv_bias", load_layer_fn),
        # Blocks MLP
        ("load_mlp_linear_fc1_weight", load_layer_fn),
        ("load_mlp_linear_fc1_bias", load_layer_fn),
        ("load_mlp_linear_fc2_weight", load_layer_fn),
        ("load_mlp_linear_fc2_bias", load_layer_fn),
        # Blocks norm
        ("load_norm1_weight", load_layer_fn),
        ("load_norm1_bias", load_layer_fn),
        ("load_norm2_weight", load_layer_fn),
        ("load_norm2_bias", load_layer_fn),
        # Deepstack merger
        ("load_deepstack_merger_linear_fc1_weight", load_layer_fn),
        ("load_deepstack_merger_linear_fc1_bias", load_layer_fn),
        ("load_deepstack_merger_linear_fc2_weight", load_layer_fn),
        ("load_deepstack_merger_linear_fc2_bias", load_layer_fn),
        ("load_deepstack_merger_norm_weight", load_layer_fn),
        ("load_deepstack_merger_norm_bias", load_layer_fn),
        # Merger
        ("load_merger_linear_fc1_weight", load_global_fn),
        ("load_merger_linear_fc1_bias", load_global_fn),
        ("load_merger_linear_fc2_weight", load_global_fn),
        ("load_merger_linear_fc2_bias", load_global_fn),
        ("load_merger_norm_weight", load_global_fn),
        ("load_merger_norm_bias", load_global_fn),
    ]


class Qwen3vlWeightLoaderCStruct(Structure):
    _fields_ = [
        ("lang_loader", Qwen3vlLangWeightLoaderCStruct),
        ("vis_loader", Qwen3vlVisWeightLoaderCStruct),
    ]


@register_model
class Qwen3vlModel(BaseModel):
    @classmethod
    def register_lib(cls, lib):
        """Register Qwen3vl model functions with the library"""
        lib.createQwen3vlWeightLoader.argtypes = []
        lib.createQwen3vlWeightLoader.restype = POINTER(Qwen3vlWeightLoaderCStruct)

        lib.createQwen3vlWeights.argtypes = [
            POINTER(Qwen3vlMetaCStruct),
            DeviceType,
            c_int,
            POINTER(c_int),
            c_bool,
        ]
        lib.createQwen3vlWeights.restype = POINTER(Qwen3vlWeightsCStruct)

        lib.createQwen3vlModel.argtypes = [
            POINTER(Qwen3vlMetaCStruct),
            POINTER(Qwen3vlWeightsCStruct),
        ]
        lib.createQwen3vlModel.restype = POINTER(Qwen3vlModelCStruct)

        lib.destroyQwen3vlModel.argtypes = [POINTER(Qwen3vlModelCStruct)]

        lib.createQwen3vlCache.argtypes = [POINTER(Qwen3vlModelCStruct)]
        lib.createQwen3vlCache.restype = POINTER(Qwen3vlCacheCStruct)

        lib.dropQwen3vlCache.argtypes = [
            POINTER(Qwen3vlModelCStruct),
            POINTER(Qwen3vlCacheCStruct),
        ]

        lib.inferBatchQwen3vl.argtypes = [
            POINTER(Qwen3vlModelCStruct),
            POINTER(c_uint),
            c_uint,
            POINTER(c_uint),
            c_uint,
            POINTER(c_uint),
            POINTER(POINTER(Qwen3vlCacheCStruct)),
            POINTER(c_float),
            POINTER(c_uint),
            POINTER(c_float),
            POINTER(c_uint),
        ]

        lib.forwardBatchQwen3vl.argtypes = [
            POINTER(Qwen3vlModelCStruct),
            POINTER(c_uint),
            c_uint,
            POINTER(c_uint),
            c_uint,
            POINTER(c_uint),
            POINTER(POINTER(Qwen3vlCacheCStruct)),
            c_void_p,
        ]

    def create_weight_loader(self):
        return self.lib.createQwen3vlWeightLoader()

    def create_weights(self, meta, device_type, ndev, dev_ids, transpose_weight):
        return self.lib.createQwen3vlWeights(meta, device_type, ndev, dev_ids, transpose_weight)

    def create_model(self, meta, weights):
        return self.lib.createQwen3vlModel(meta, weights)

    def destroy_model(self, model):
        self.lib.destroyQwen3vlModel(model)

    def create_cache(self, model):
        return self.lib.createQwen3vlCache(model)

    def drop_cache(self, model, cache):
        self.lib.dropQwen3vlCache(model, cache)

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
        self.lib.inferBatchQwen3vl(
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

    def forward_batch(
        self,
        model,
        tokens,
        ntok,
        req_lens,
        nreq,
        req_pos,
        caches,
        logits,
    ):
        self.lib.forwardBatchQwen3vl(
            model,
            tokens,
            ntok,
            req_lens,
            nreq,
            req_pos,
            caches,
            logits,
        )