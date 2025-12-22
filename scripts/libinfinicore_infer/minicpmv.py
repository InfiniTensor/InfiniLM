from .base import BaseModel, DataType, DeviceType, register_model
from ctypes import c_size_t, c_uint, c_int, c_float, c_void_p, POINTER, Structure


class MiniCPMVVisionMetaCStruct(Structure):
    _fields_ = [
        ("patch_size", c_size_t),
        ("vision_embed_dim", c_size_t),
        ("vision_num_layers", c_size_t),
        ("vision_num_heads", c_size_t),
        ("vision_intermediate_size", c_size_t),
        ("vision_layer_norm_eps", c_float),
        ("vision_image_size", c_size_t),
        ("vision_num_positions", c_size_t),
    ]


class MiniCPMVResamplerMetaCStruct(Structure):
    _fields_ = [
        ("num_queries", c_size_t),
        ("embed_dim", c_size_t),
        ("num_heads", c_size_t),
        ("kv_dim", c_size_t),
        ("layer_norm_eps", c_float),
        ("max_patches_h", c_size_t),
        ("max_patches_w", c_size_t),
    ]


class MiniCPMVLanguageMetaCStruct(Structure):
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


class MiniCPMVMetaCStruct(Structure):
    _fields_ = [
        ("vision_meta", MiniCPMVVisionMetaCStruct),
        ("resampler_meta", MiniCPMVResamplerMetaCStruct),
        ("language_meta", MiniCPMVLanguageMetaCStruct),
    ]


class MiniCPMVSiglipLayerWeightsCStruct(Structure):
    _fields_ = [
        ("layer_norm1_weight", c_void_p),
        ("layer_norm1_bias", c_void_p),
        ("layer_norm2_weight", c_void_p),
        ("layer_norm2_bias", c_void_p),
        ("q_weight", c_void_p),
        ("q_bias", c_void_p),
        ("k_weight", c_void_p),
        ("k_bias", c_void_p),
        ("v_weight", c_void_p),
        ("v_bias", c_void_p),
        ("out_weight", c_void_p),
        ("out_bias", c_void_p),
        ("fc1_weight", c_void_p),
        ("fc1_bias", c_void_p),
        ("fc2_weight", c_void_p),
        ("fc2_bias", c_void_p),
    ]


class MiniCPMVWeightsCStruct(Structure):
    _fields_ = [
        # Vision
        ("vpm_patch_embedding_weight", c_void_p),
        ("vpm_patch_embedding_bias", c_void_p),
        ("vpm_position_embedding", c_void_p),
        ("vpm_layers", POINTER(MiniCPMVSiglipLayerWeightsCStruct)),
        ("vpm_post_layernorm_weight", c_void_p),
        ("vpm_post_layernorm_bias", c_void_p),
        # Resampler
        ("resampler_query", c_void_p),
        ("resampler_kv_proj_weight", c_void_p),
        ("resampler_attn_in_proj_weight", c_void_p),
        ("resampler_attn_in_proj_bias", c_void_p),
        ("resampler_attn_out_proj_weight", c_void_p),
        ("resampler_attn_out_proj_bias", c_void_p),
        ("resampler_ln_q_weight", c_void_p),
        ("resampler_ln_q_bias", c_void_p),
        ("resampler_ln_kv_weight", c_void_p),
        ("resampler_ln_kv_bias", c_void_p),
        ("resampler_ln_post_weight", c_void_p),
        ("resampler_ln_post_bias", c_void_p),
        ("resampler_proj", c_void_p),
        # Language (reuse Jiuge layout)
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


class MiniCPMVModelCStruct(Structure):
    pass


@register_model
class MiniCPMVModel(BaseModel):
    @classmethod
    def register_lib(cls, lib):
        lib.createMiniCPMVModel.restype = POINTER(MiniCPMVModelCStruct)
        lib.createMiniCPMVModel.argtypes = [
            POINTER(MiniCPMVMetaCStruct),
            POINTER(MiniCPMVWeightsCStruct),
            DeviceType,
            c_int,
            POINTER(c_int),
        ]

        lib.destroyMiniCPMVModel.argtypes = [POINTER(MiniCPMVModelCStruct)]

        lib.inferMiniCPMVResampler.argtypes = [
            POINTER(MiniCPMVModelCStruct),
            c_void_p,   # x
            c_size_t,   # seq_len
            c_uint,     # tgt_h
            c_uint,     # tgt_w
            c_void_p,   # output
        ]
        lib.inferMiniCPMVResampler.restype = None

        lib.inferMiniCPMVSiglipEmbeddings.argtypes = [
            POINTER(MiniCPMVModelCStruct),
            c_void_p,  # pixel_values
            c_size_t,  # seq_len
            c_uint,  # tgt_h
            c_uint,  # tgt_w
            c_void_p,  # output
        ]
        lib.inferMiniCPMVSiglipEmbeddings.restype = None

        lib.inferMiniCPMVSiglipLayer0.argtypes = [
            POINTER(MiniCPMVModelCStruct),
            c_void_p,  # hidden_states
            c_size_t,  # seq_len
            c_void_p,  # output
        ]
        lib.inferMiniCPMVSiglipLayer0.restype = None

        lib.inferMiniCPMVSiglipLayer.argtypes = [
            POINTER(MiniCPMVModelCStruct),
            c_uint,   # layer_idx
            c_void_p,  # hidden_states
            c_size_t,  # seq_len
            c_void_p,  # output
        ]
        lib.inferMiniCPMVSiglipLayer.restype = None

        lib.inferMiniCPMVSiglipEncoder.argtypes = [
            POINTER(MiniCPMVModelCStruct),
            c_uint,   # num_layers
            c_void_p,  # hidden_states
            c_size_t,  # seq_len
            c_void_p,  # output
        ]
        lib.inferMiniCPMVSiglipEncoder.restype = None

        lib.inferMiniCPMVVisionResampler.argtypes = [
            POINTER(MiniCPMVModelCStruct),
            c_void_p,  # pixel_values
            c_size_t,  # seq_len
            c_uint,  # tgt_h
            c_uint,  # tgt_w
            c_void_p,  # output
        ]
        lib.inferMiniCPMVVisionResampler.restype = None

    def create_model(self, meta, weights, device_type, ndev, dev_ids):
        return self.lib.createMiniCPMVModel(meta, weights, device_type, ndev, dev_ids)

    def destroy_model(self, model):
        self.lib.destroyMiniCPMVModel(model)

    def infer_resampler(self, model, x, seq_len, tgt_h, tgt_w, output):
        self.lib.inferMiniCPMVResampler(model, x, seq_len, tgt_h, tgt_w, output)

    def infer_siglip_embeddings(self, model, pixel_values, seq_len, tgt_h, tgt_w, output):
        self.lib.inferMiniCPMVSiglipEmbeddings(
            model, pixel_values, seq_len, tgt_h, tgt_w, output
        )

    def infer_siglip_layer0(self, model, hidden_states, seq_len, output):
        self.lib.inferMiniCPMVSiglipLayer0(model, hidden_states, seq_len, output)

    def infer_siglip_layer(self, model, layer_idx, hidden_states, seq_len, output):
        self.lib.inferMiniCPMVSiglipLayer(
            model, layer_idx, hidden_states, seq_len, output
        )

    def infer_siglip_encoder(self, model, num_layers, hidden_states, seq_len, output):
        self.lib.inferMiniCPMVSiglipEncoder(
            model, num_layers, hidden_states, seq_len, output
        )

    def infer_vision_resampler(self, model, pixel_values, seq_len, tgt_h, tgt_w, output):
        self.lib.inferMiniCPMVVisionResampler(
            model, pixel_values, seq_len, tgt_h, tgt_w, output
        )
