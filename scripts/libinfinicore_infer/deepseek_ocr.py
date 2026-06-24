from .base import BaseModel, DataType, DeviceType, KVCacheCStruct, register_model
from ctypes import c_size_t, c_uint, c_int, c_float, c_void_p, POINTER, Structure, byref


class DeepSeekOCRMetaCStruct(Structure):
    _fields_ = [
        ("dt_logits", DataType),
        ("dt_norm", DataType),
        # Layer counts
        ("n_dense_layer", c_size_t),  # 第0层是dense
        ("n_sparse_layer", c_size_t),  # 第1-11层是MoE
        # Model dimensions
        ("d", c_size_t),  # hidden_size: 1280
        ("nh", c_size_t),  # num_attention_heads: 1280
        ("nkvh", c_size_t),  # num_key_value_heads: 1280
        ("dh", c_size_t),  # head_dim: d/nh = 1
        # Dense MLP dimensions
        ("di_dense", c_size_t),  # intermediate_size for dense layer: 6848
        # MoE dimensions
        ("di_moe", c_size_t),  # moe_intermediate_size: 896
        ("di_shared", c_size_t),  # shared_expert_intermediate_size: 1792
        ("nexperts", c_size_t),  # n_routed_experts: 64
        ("kexperts", c_size_t),  # num_experts_per_tok: 6
        ("routed_scale", c_float),  # routed_scaling_factor: 1.0
        # Context and vocab
        ("dctx", c_size_t),  # max_position_embeddings
        ("dvoc", c_size_t),  # vocab_size: 129280
        # Normalization
        ("epsilon", c_float),  # rms_norm_eps: 1e-6
        ("theta", c_float),  # rope_theta: 10000.0
        ("end_token", c_uint),  # eos_token_id
    ]


class DeepSeekOCRWeightsCStruct(Structure):
    _fields_ = [
        ("n_dense_layer", c_size_t),
        ("n_sparse_layer", c_size_t),
        ("dt_norm", DataType),
        ("dt_mat", DataType),
        ("transpose_linear_weights", c_int),
        # Embeddings
        ("input_embd", c_void_p),  # [dvoc, d]
        ("output_norm", c_void_p),  # [d]
        ("output_embd", c_void_p),  # [dvoc, d]
        # Attention layers (all layers)
        ("attn_norm", POINTER(c_void_p)),  # nlayer * [d]
        ("attn_q", POINTER(c_void_p)),  # nlayer * [d, d] 或分片
        ("attn_k", POINTER(c_void_p)),  # nlayer * [d, d] 或分片
        ("attn_v", POINTER(c_void_p)),  # nlayer * [d, d] 或分片
        ("attn_o", POINTER(c_void_p)),  # nlayer * [d, d] 或分片
        # FFN layers
        ("ffn_norm", POINTER(c_void_p)),  # nlayer * [d]
        # Dense MLP (layer 0)
        ("dense_gate", c_void_p),  # [di_dense, d]
        ("dense_up", c_void_p),  # [di_dense, d]
        ("dense_down", c_void_p),  # [d, di_dense]
        # MoE layers (layer 1-11)
        ("moe_gate_weight", POINTER(c_void_p)),  # n_sparse_layer * [nexperts, d]
        ("moe_gate_bias", POINTER(c_void_p)),  # n_sparse_layer * [nexperts]
        # Shared experts
        ("moe_shared_gate", POINTER(c_void_p)),  # n_sparse_layer * [di_shared, d]
        ("moe_shared_up", POINTER(c_void_p)),  # n_sparse_layer * [di_shared, d]
        ("moe_shared_down", POINTER(c_void_p)),  # n_sparse_layer * [d, di_shared]
        # Routed experts
        ("moe_experts_gate", POINTER(POINTER(c_void_p))),  # n_sparse_layer * nexperts * [di_moe, d]
        ("moe_experts_up", POINTER(POINTER(c_void_p))),  # n_sparse_layer * nexperts * [di_moe, d]
        ("moe_experts_down", POINTER(POINTER(c_void_p))),  # n_sparse_layer * nexperts * [d, di_moe]
        # Vision weights - SAM
        ("sam_patch_embed", c_void_p),
        ("sam_patch_embed_bias", c_void_p),
        ("sam_block_norm1", POINTER(c_void_p)),  # 12 layers
        ("sam_block_attn_qkv", POINTER(c_void_p)),
        ("sam_block_attn_proj", POINTER(c_void_p)),
        ("sam_block_norm2", POINTER(c_void_p)),
        ("sam_block_mlp_fc1", POINTER(c_void_p)),
        ("sam_block_mlp_fc2", POINTER(c_void_p)),
        ("sam_neck_conv1", c_void_p),
        ("sam_neck_ln1", c_void_p),
        ("sam_neck_conv2", c_void_p),
        ("sam_neck_ln2", c_void_p),
        # Vision weights - CLIP
        ("clip_patch_embed", c_void_p),
        ("clip_patch_embed_bias", c_void_p),
        ("clip_position_embed", c_void_p),
        ("clip_pre_layernorm", c_void_p),
        ("clip_block_ln1", POINTER(c_void_p)),  # 24 layers
        ("clip_block_attn_qkv", POINTER(c_void_p)),
        ("clip_block_attn_proj", POINTER(c_void_p)),
        ("clip_block_ln2", POINTER(c_void_p)),
        ("clip_block_mlp_fc1", POINTER(c_void_p)),
        ("clip_block_mlp_fc2", POINTER(c_void_p)),
        # Projector
        ("projector", c_void_p),  # [2048, 1280]
        ("image_newline", c_void_p),  # [1280]
        ("view_seperator", c_void_p),  # [1280]
    ]


class DeepSeekOCRModelCStruct(Structure):
    pass


@register_model
class DeepSeekOCRModel(BaseModel):
    @classmethod
    def register_lib(cls, lib):
        lib.createDeepSeekOCRModel.restype = POINTER(DeepSeekOCRModelCStruct)
        lib.createDeepSeekOCRModel.argtypes = [
            POINTER(DeepSeekOCRMetaCStruct),
            POINTER(DeepSeekOCRWeightsCStruct),
            DeviceType,
            c_int,
            POINTER(c_int),
        ]

        lib.destroyDeepSeekOCRModel.argtypes = [POINTER(DeepSeekOCRModelCStruct)]

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

        lib.inferBatchDeepSeekOCR.argtypes = [
            POINTER(DeepSeekOCRModelCStruct),
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

        lib.forwardBatchDeepSeekOCR.argtypes = [
            POINTER(DeepSeekOCRModelCStruct),
            POINTER(c_uint),
            c_uint,
            POINTER(c_uint),
            c_uint,
            POINTER(c_uint),
            POINTER(POINTER(KVCacheCStruct)),
            c_void_p,
        ]

        # 新增: 用于注入图像特征的接口
        lib.inferBatchDeepSeekOCRWithEmbeds.argtypes = [
            POINTER(DeepSeekOCRModelCStruct),
            c_void_p,  # inputs_embeds
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

    def create_model(self, meta, weights, device_type, ndev, dev_ids):
        return self.lib.createDeepSeekOCRModel(meta, weights, device_type, ndev, dev_ids)

    def destroy_model(self, model):
        self.lib.destroyDeepSeekOCRModel(model)

    def create_kv_cache(
        self, nlayer, max_len, nkvh, dk, dv, dtype, device, dev_ids, ndev
    ):
        return self.lib.createKVCache(
            nlayer, max_len, nkvh, dk, dv, dtype, device, dev_ids, ndev
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
        temperature,
        topk,
        topp,
        output,
    ):
        self.lib.inferBatchDeepSeekOCR(
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
        self.lib.forwardBatchDeepSeekOCR(
            model, tokens, ntok, req_lens, nreq, req_pos, kv_caches, logits
        )

    def infer_batch_with_embeds(
        self,
        model,
        inputs_embeds,
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
        self.lib.inferBatchDeepSeekOCRWithEmbeds(
            model,
            inputs_embeds,
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
