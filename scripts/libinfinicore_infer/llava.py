from .base import BaseModel, DataType, DeviceType, KVCacheCStruct, register_model
from ctypes import c_size_t, c_uint, c_int, c_float, c_void_p, POINTER, Structure, byref


class LlavaVisionMetaCStruct(Structure):
    _fields_ = [
        ("image_size", c_size_t),
        ("patch_size", c_size_t),
        ("num_patches", c_size_t),
        ("vision_embed_dim", c_size_t),
        ("vision_num_layers", c_size_t),
        ("vision_num_heads", c_size_t),
        ("vision_intermediate_size", c_size_t),
        ("vision_epsilon", c_float),
    ]


class LlavaLanguageMetaCStruct(Structure):
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


class LlavaProjectorMetaCStruct(Structure):
    _fields_ = [
        ("vision_embed_dim", c_size_t),
        ("text_embed_dim", c_size_t),
        ("projector_hidden_size", c_size_t),
    ]


class LlavaMetaCStruct(Structure):
    _fields_ = [
        ("vision_meta", LlavaVisionMetaCStruct),
        ("language_meta", LlavaLanguageMetaCStruct),
        ("projector_meta", LlavaProjectorMetaCStruct),
    ]


class LlavaWeightsCStruct(Structure):
    _fields_ = [
        # Vision Encoder Weights
        ("vision_nlayer", c_void_p),
        ("vision_patch_embed_weight", c_void_p),
        ("vision_class_token", c_void_p),
        ("vision_position_embedding", c_void_p),
        ("vision_encoder_weights", POINTER(c_void_p)), # 好像没用
        ("vision_pre_layernorm_weight", c_void_p),
        ("vision_pre_layernorm_bias", c_void_p),
        ("vision_post_layernorm_weight", c_void_p),
        ("vision_post_layernorm_bias", c_void_p),

        ("vision_q_weights", POINTER(c_void_p)),
        ("vision_q_biases", POINTER(c_void_p)),
        ("vision_k_weights", POINTER(c_void_p)),
        ("vision_k_biases", POINTER(c_void_p)),
        ("vision_v_weights", POINTER(c_void_p)),
        ("vision_v_biases", POINTER(c_void_p)),

        ("vision_in_layer_pre_norm_weights", POINTER(c_void_p)),
        ("vision_in_layer_pre_norm_biases", POINTER(c_void_p)),

        ("vision_proj_weight", POINTER(c_void_p)),
        ("vision_proj_bias", POINTER(c_void_p)),

        ("vision_in_layer_post_norm_weight", POINTER(c_void_p)),
        ("vision_post_norm_bias", POINTER(c_void_p)),

        ("vision_mlp_fc1_weight", POINTER(c_void_p)),
        ("vision_mlp_fc1_bias", POINTER(c_void_p)),

        ("vision_mlp_fc2_weight", POINTER(c_void_p)),
        ("vision_mlp_fc2_bias", POINTER(c_void_p)),



        # MultiModal Projector Weights
        ("projector_weight_1", c_void_p),
        ("projector_bias_1", c_void_p),
        ("projector_weight_2", c_void_p),
        ("projector_bias_2", c_void_p),

        # Language Model Weights (reuse Jiuge structure)
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


class LlavaKVCacheCStruct(Structure):
    _fields_ = [
        ("past_key", c_void_p),
        ("past_value", c_void_p),
        ("past_seq_len", c_size_t),
        ("max_seq_len", c_size_t),
    ]

class LlavaModelCStruct(Structure):
    pass

@register_model
class LlavaModel(BaseModel):
    def __init__(self):
        super().__init__()

    @classmethod
    def register_lib(cls, lib):
        # Setup function signatures
        lib.createLlavaModel.restype = POINTER(LlavaModelCStruct)
        lib.createLlavaModel.argtypes = [
            POINTER(LlavaMetaCStruct),
            POINTER(LlavaWeightsCStruct),
            DeviceType,  # device
            c_int,  # ndev
            POINTER(c_int),  # dev_ids
        ]

        lib.destroyLlavaModel.argtypes = [POINTER(LlavaModelCStruct)]

        lib.createKVCache.argtypes = [
            c_size_t,  # nlayer
            c_size_t,  # max_len
            c_size_t,  # nkvh
            c_size_t,  # dk
            c_size_t,  # dv
            DataType,  # dtype
            DeviceType,  # device
            POINTER(c_int),  # dev_ids
            c_size_t,  # ndev
        ]
        lib.createKVCache.restype = POINTER(KVCacheCStruct)
        lib.dropKVCache.argtypes = [POINTER(KVCacheCStruct)]

        # 新增：LLaVA Vision Encoding (用于batch_infer_vision)
        lib.inferBatchLlavaVison.argtypes = [
            POINTER(LlavaModelCStruct),  # model
            c_void_p,  # image_data
            c_void_p,  # output
        ]
        lib.inferBatchLlavaVison.restype = None

        # lib.encodeVision.argtypes = [
        #     POINTER(c_void_p),  # model
        #     c_void_p,  # image_tensor
        #     c_void_p,  # output
        # ]
        # lib.encodeVision.restype = None

        # lib.projectMultiModal.argtypes = [
        #     POINTER(c_void_p),  # model
        #     c_void_p,  # vision_features
        #     c_void_p,  # output
        # ]
        # lib.projectMultiModal.restype = None

        # lib.inferBatchLlavaLanguage.argtypes = [
        #     POINTER(c_void_p),  # model
        #     POINTER(c_uint),  # tokens
        #     c_size_t,  # ntok
        #     POINTER(c_size_t),  # req_lens
        #     c_size_t,  # nreq
        #     POINTER(c_size_t),  # req_pos
        #     POINTER(LlavaKVCacheCStruct),  # kv_caches
        #     POINTER(c_float),  # temperature
        #     POINTER(c_float),  # topk
        #     POINTER(c_float),  # topp
        #     POINTER(c_uint),  # output
        # ]
        # lib.inferBatchLlavaLanguage.restype = None

    def create_model(self, meta, weights, device, ndev, dev_ids):
        return self.lib.createLlavaModel(
            meta,
            weights,
            device,
            ndev,
            dev_ids,
        )

    def destroy_model(self, model):
        self.lib.destroyLlavaModel(model)

    def create_kv_cache(
        self, nlayer, max_len, nkvh, dk, dv, dtype, device, dev_ids, ndev
    ):
        return self.lib.createKVCache(
            nlayer, max_len, nkvh, dk, dv, dtype, device, dev_ids, ndev
        )

    def drop_kv_cache(self, kv_cache):
        self.lib.dropKVCache(kv_cache)

    def infer_batch_vision(self, model, image_data, output):
        """LLaVA Vision Encoding - 对应Python中的infer_batch_vision"""
        self.lib.inferBatchLlavaVison(model, image_data, output)

    def encode_vision(self, model, image_tensor, output):
        self.lib.encodeVision(model, image_tensor, output)

    def project_multimodal(self, model, vision_features, output):
        self.lib.projectMultiModal(model, vision_features, output)

    def infer_batch_language(self, model, tokens, ntok, req_lens, nreq, req_pos,
                           kv_caches, temperature, topk, topp, output):
        self.lib.inferBatchLlavaLanguage(
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
