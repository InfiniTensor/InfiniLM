from typing import List, Sequence
import math
import os
from pathlib import Path
import safetensors
import sys
import time
import json
import torch
import transformers
from transformers import AutoProcessor
import ctypes
from ctypes import c_int, c_void_p, c_uint, byref
import numpy as np
# from PIL import Image
# import numpy as np


from libinfinicore_infer import (
    JiugeMetaCStruct,
    JiugeWeightsCStruct,
    LlavaMetaCStruct,
    LlavaVisionMetaCStruct,
    LlavaLanguageMetaCStruct,
    LlavaProjectorMetaCStruct,
    LlavaWeightsCStruct,
    LlavaModel,
    DataType,
    DeviceType,
)
from infer_task import InferTask, KVCache



class LlamaWeightsNaming:
    def input_embd(self):
        return "language_model.model.embed_tokens.weight"

    def output_norm(self):
        return "language_model.model.norm.weight"

    def output_embd(self):
        return "language_model.lm_head.weight"

    def attn_norm(self, i):
        return f"language_model.model.layers.{i}.input_layernorm.weight"

    def attn_q(self, i):
        return f"language_model.model.layers.{i}.self_attn.q_proj.weight"

    def attn_k(self, i):
        return f"language_model.model.layers.{i}.self_attn.k_proj.weight"

    def attn_v(self, i):
        return f"language_model.model.layers.{i}.self_attn.v_proj.weight"

    def attn_o(self, i):
        return f"language_model.model.layers.{i}.self_attn.o_proj.weight"

    def attn_q_b(self, i):
        return f"language_model.model.layers.{i}.self_attn.q_proj.bias"

    def attn_k_b(self, i):
        return f"language_model.model.layers.{i}.self_attn.k_proj.bias"

    def attn_v_b(self, i):
        return f"model.layers.{i}.self_attn.v_proj.bias"

    def attn_q_norm(self, i):
        return f"language_model.model.layers.{i}.self_attn.q_norm.weight"

    def attn_k_norm(self, i):
        return f"language_model.model.layers.{i}.self_attn.k_norm.weight"

    def ffn_norm(self, i):
        return f"language_model.model.layers.{i}.post_attention_layernorm.weight"

    def gate(self, i):
        return f"language_model.model.layers.{i}.mlp.gate_proj.weight"

    def up(self, i):
        return f"language_model.model.layers.{i}.mlp.up_proj.weight"

    def down(self, i):
        return f"language_model.model.layers.{i}.mlp.down_proj.weight"

    def match(state_dict):
        return (
            "model.norm.weight" in state_dict
            and "model.layers.0.self_attn.q_proj.weight" in state_dict
        )



class JiugeMetaFromLlama(JiugeMetaCStruct):
    def __init__(self, config, dtype=torch.float16, max_tokens=None):
        if dtype == torch.float16:
            dt_ = DataType.INFINI_DTYPE_F16
        elif dtype == torch.float32:
            dt_ = DataType.INFINI_DTYPE_F32
        elif dtype == torch.bfloat16:
            dt_ = DataType.INFINI_DTYPE_BF16
        else:
            dt_ = DataType.INFINI_DTYPE_F16

        self.scale_input = 1.0
        self.scale_output = 1.0
        self.scale_o = 1.0
        self.scale_down = 1.0
        if (
            config["model_type"] in ["fm9g", "minicpm"]
            and "scale_emb" in config
            and "scale_depth" in config
            and "dim_model_base" in config
        ):
            self.scale_input = config["scale_emb"]
            self.scale_output = config["hidden_size"] // config["dim_model_base"]
            self.scale_o = config["scale_depth"] / math.sqrt(
                config["num_hidden_layers"]
            )
            self.scale_down = config["scale_depth"] / math.sqrt(
                config["num_hidden_layers"]
            )

        super().__init__(
            dt_logits=dt_,
            # nlayer=config["num_hidden_layers"],
            nlayer=32,      # vicuna-7b-v1.5 config
            d=4096,
            nh=32,
            nkvh=32,
            dh=(4096 // 32),
            di=11008,
            dctx=(
                4096 if max_tokens is None else max_tokens
            ),
            dvoc=32064,
            epsilon=1e-05,
            theta=(config["rope_theta"] if "rope_theta" in config else 100000.0),
            end_token=2,
        )
        self.torch_dtype_logits = dtype




class JiugeWeightsImpl(JiugeWeightsCStruct):
    def __init__(
        self,
        meta,
        naming,
        state_dict,
        torch_dt_mat=torch.float16,
        torch_dt_norm=torch.float32,
        ndev=1,
        transpose_weight=True,
    ):
        nlayer = meta.nlayer
        nh = meta.nh
        nkvh = meta.nkvh
        dh = meta.dh
        d = meta.d
        di = meta.di
        scale_input = meta.scale_input
        scale_output = meta.scale_output
        scale_o = meta.scale_o
        scale_down = meta.scale_down
        assert nh % nkvh == 0
        assert nh % ndev == 0
        assert nkvh % ndev == 0
        assert di % ndev == 0
        torch_dt_logits = meta.torch_dtype_logits
        if torch_dt_mat == torch.float16:
            self.dt_mat = DataType.INFINI_DTYPE_F16
        elif torch_dt_mat == torch.float32:
            self.dt_mat = DataType.INFINI_DTYPE_F32
        elif torch_dt_mat == torch.bfloat16:
            self.dt_mat = DataType.INFINI_DTYPE_BF16
        else:
            raise ValueError("Unsupported proj weight data type")
        if torch_dt_norm == torch.float16:
            self.dt_norm = DataType.INFINI_DTYPE_F16
        elif torch_dt_norm == torch.float32:
            self.dt_norm = DataType.INFINI_DTYPE_F32
        elif torch_dt_norm == torch.bfloat16:
            self.dt_norm = DataType.INFINI_DTYPE_BF16
        else:
            raise ValueError("Unsupported norm weight data type")

        input_embd_naming = (
            naming.input_embd()
            if naming.input_embd() in state_dict
            else naming.output_embd()
        )
        output_embd_naming = (
            naming.output_embd()
            if naming.output_embd() in state_dict
            else naming.input_embd()
        )
        self.transpose_linear_weights = 1 if transpose_weight else 0
        self.nlayer = nlayer
        self.input_embd_tensor = (
            state_dict[input_embd_naming].to(torch_dt_logits) * scale_input
        )
        self.input_embd = self.input_embd_tensor.data_ptr()
        self.output_norm_tensor = (
            state_dict[naming.output_norm()].to(torch_dt_norm) * scale_output
        )
        self.output_norm = self.output_norm_tensor.data_ptr()
        self.output_embd_tensor = state_dict[output_embd_naming].to(torch_dt_mat)
        if not transpose_weight:
            self.output_embd_tensor = self.output_embd_tensor.transpose(
                0, 1
            ).contiguous()
        self.output_embd = self.output_embd_tensor.data_ptr()

        self.attn_norm_tensors = [
            state_dict[naming.attn_norm(i)].to(torch_dt_norm) for i in range(nlayer)
        ]
        self.attn_norm_ptrs = [
            self.attn_norm_tensors[i].data_ptr() for i in range(nlayer)
        ]
        self.attn_norm = (c_void_p * nlayer)(*self.attn_norm_ptrs)

        def qkv_slices(_i):
            _Q = (
                state_dict[naming.attn_q(_i)]
                .reshape([nh, 2, dh // 2, d])
                .transpose(1, 2)
            )
            _K = (
                state_dict[naming.attn_k(_i)]
                .reshape([nkvh, 2, dh // 2, d])
                .transpose(1, 2)
            )
            _V = state_dict[naming.attn_v(_i)].reshape([nkvh, dh // 2, 2, d])
            _result = []
            _nh = nh // ndev
            _nkvh = nkvh // ndev
            for _idev in range(ndev):
                _result.append(_Q[_idev * _nh : (_idev + 1) * _nh, :, :, :])
                _result.append(_K[_idev * _nkvh : (_idev + 1) * _nkvh, :, :, :])
                _result.append(_V[_idev * _nkvh : (_idev + 1) * _nkvh, :, :])
            return _result

        self.qkv_tensor = [
            torch.concat(qkv_slices(i)).to(torch_dt_mat) for i in range(nlayer)
        ]
        if not transpose_weight:
            for i in range(nlayer):
                self.qkv_tensor[i] = (
                    self.qkv_tensor[i]
                    .reshape(ndev, (nh + 2 * nkvh) // ndev * dh, d)
                    .transpose(1, 2)
                    .contiguous()
                )
        self.qkv_tensor_ptrs = [self.qkv_tensor[i].data_ptr() for i in range(nlayer)]
        self.attn_qkv = (c_void_p * nlayer)(*self.qkv_tensor_ptrs)

        def qkv_b_slices(_i):
            _QB = (
                state_dict[naming.attn_q_b(_i)]
                .reshape([nh, 2, dh // 2])
                .transpose(1, 2)
            )
            _KB = (
                state_dict[naming.attn_k_b(_i)]
                .reshape([nkvh, 2, dh // 2])
                .transpose(1, 2)
            )
            _VB = state_dict[naming.attn_v_b(_i)].reshape([nkvh, dh // 2, 2])
            _result = []
            _nh = nh // ndev
            _nkvh = nkvh // ndev
            for _idev in range(ndev):
                _result.append(_QB[_idev * _nh : (_idev + 1) * _nh, :, :].flatten())
                _result.append(_KB[_idev * _nkvh : (_idev + 1) * _nkvh, :, :].flatten())
                _result.append(_VB[_idev * _nkvh : (_idev + 1) * _nkvh, :, :].flatten())
            return _result

        if naming.attn_q_b(0) in state_dict:
            self.qkv_b_tensors = [
                torch.concat(qkv_b_slices(i)).to(torch_dt_logits) for i in range(nlayer)
            ]
            self.qkv_b_tensor_ptrs = [
                self.qkv_b_tensors[i].data_ptr() for i in range(nlayer)
            ]
            self.attn_qkv_b = (c_void_p * nlayer)(*self.qkv_b_tensor_ptrs)
        else:
            self.attn_qkv_b = None

        if naming.attn_q_norm(0) in state_dict:
            self.attn_q_norm_tensors = [
                state_dict[naming.attn_q_norm(i)]
                .reshape([2, dh // 2])
                .transpose(0, 1)
                .contiguous()
                .to(torch_dt_norm)
                for i in range(nlayer)
            ]
            self.attn_q_norm_ptrs = [
                self.attn_q_norm_tensors[i].data_ptr() for i in range(nlayer)
            ]
            self.attn_q_norm = (c_void_p * nlayer)(*self.attn_q_norm_ptrs)
            self.attn_k_norm_tensors = [
                state_dict[naming.attn_k_norm(i)]
                .reshape([2, dh // 2])
                .transpose(0, 1)
                .contiguous()
                .to(torch_dt_norm)
                for i in range(nlayer)
            ]
            self.attn_k_norm_ptrs = [
                self.attn_k_norm_tensors[i].data_ptr() for i in range(nlayer)
            ]
            self.attn_k_norm = (c_void_p * nlayer)(*self.attn_k_norm_ptrs)
        else:
            self.attn_q_norm = None
            self.attn_k_norm = None

        self.attn_o_tensor = [
            (
                state_dict[naming.attn_o(i)]
                .to(torch_dt_mat)
                .reshape([d, ndev, nh // ndev * dh])
                .transpose(0, 1)
                .contiguous()
                if transpose_weight
                else state_dict[naming.attn_o(i)]
                .transpose(0, 1)
                .to(torch_dt_mat)
                .contiguous()
            )
            * scale_o
            for i in range(nlayer)
        ]
        self.attn_o_ptrs = [self.attn_o_tensor[i].data_ptr() for i in range(nlayer)]
        self.attn_o = (c_void_p * nlayer)(*self.attn_o_ptrs)

        self.ffn_norm_tensors = [
            state_dict[naming.ffn_norm(i)].to(torch_dt_norm) for i in range(nlayer)
        ]
        self.ffn_norm_ptrs = [
            self.ffn_norm_tensors[i].data_ptr() for i in range(nlayer)
        ]
        self.ffn_norm = (c_void_p * nlayer)(*self.ffn_norm_ptrs)

        def gate_up_slices(_i):
            _result = []
            _di = di // ndev
            for _idev in range(ndev):
                _start = _idev * _di
                _end = (_idev + 1) * _di
                _result.append(state_dict[naming.gate(_i)][_start:_end, :])
                _result.append(state_dict[naming.up(_i)][_start:_end, :])
            return _result

        self.gate_up_tensors = [
            torch.concat(gate_up_slices(i)).to(torch_dt_mat) for i in range(nlayer)
        ]
        if not transpose_weight:
            for i in range(nlayer):
                self.gate_up_tensors[i] = (
                    self.gate_up_tensors[i]
                    .reshape(ndev, 2 * di // ndev, d)
                    .transpose(1, 2)
                    .contiguous()
                )
        self.gate_up_ptrs = [self.gate_up_tensors[i].data_ptr() for i in range(nlayer)]
        self.ffn_gate_up = (c_void_p * nlayer)(*self.gate_up_ptrs)

        self.ffn_down_tensor = [
            (
                state_dict[naming.down(i)]
                .to(torch_dt_mat)
                .reshape([d, ndev, di // ndev])
                .transpose(0, 1)
                .contiguous()
                if transpose_weight
                else state_dict[naming.down(i)]
                .transpose(0, 1)
                .to(torch_dt_mat)
                .contiguous()
            )
            * scale_down
            for i in range(nlayer)
        ]
        self.ffn_down_ptrs = [self.ffn_down_tensor[i].data_ptr() for i in range(nlayer)]
        self.ffn_down = (c_void_p * nlayer)(*self.ffn_down_ptrs)




class LlavaWeightsNaming:
    """LLaVA权重命名映射类"""

    def input_embd(self):
        """输入嵌入层权重名"""
        return "language_model.model.embed_tokens.weight"

    def output_norm(self):
        """输出层归一化权重名"""
        return "language_model.model.norm.weight"

    def output_embd(self):
        """输出嵌入层权重名"""
        return "language_model.lm_head.weight"

    def vision_patch_embed_weight(self):
        """视觉编码器patch嵌入权重名"""
        return "vision_tower.vision_model.embeddings.patch_embedding.weight"

    def vision_position_embedding(self):
        """视觉编码器位置嵌入权重名"""
        return "vision_tower.vision_model.embeddings.position_embedding.weight"

    # def vision_class_embedding(self):
    #     return "vision_tower.vision_model.embeddings.class_embedding.weight"

    def vision_class_token(self):
        """视觉编码器class token权重名"""
        return "vision_tower.vision_model.embeddings.class_embedding"

    def vision_post_layernorm_bias(self):
        """视觉编码器 post_layernorm.bias 权重名"""
        return "vision_tower.vision_model.post_layernorm.bias"

    def vision_post_layernorm_weight(self):
        """视觉编码器 post_layernorm.weight 权重名"""
        return "vision_tower.vision_model.post_layernorm.weight"

    def vision_pre_layernorm_bias(self):
        """视觉编码器 pre_layernorm.bias 权重名"""
        return "vision_tower.vision_model.pre_layrnorm.bias"

    def vision_pre_layernorm_weight(self):
        """视觉编码器 pre_layernorm.weight 权重名"""
        return "vision_tower.vision_model.pre_layrnorm.weight"

    def vision_in_layer_pre_norm_weights(self, layer_idx):
        """视觉编码器前置归一化权重名"""
        return f"vision_tower.vision_model.encoder.layers.{layer_idx}.layer_norm1.weight"

    def vision_in_layer_pre_norm_biases(self, layer_idx):
        """视觉编码器前置归一化偏置名"""
        return f"vision_tower.vision_model.encoder.layers.{layer_idx}.layer_norm1.bias"

    def vision_q_weights(self, layer_idx):
        """视觉编码器Q权重名"""
        return f"vision_tower.vision_model.encoder.layers.{layer_idx}.self_attn.q_proj.weight"

    def vision_q_biases(self, layer_idx):
        """视觉编码器Q偏置名"""
        return f"vision_tower.vision_model.encoder.layers.{layer_idx}.self_attn.q_proj.bias"

    def vision_k_weights(self, layer_idx):
        """视觉编码器K权重名"""
        return f"vision_tower.vision_model.encoder.layers.{layer_idx}.self_attn.k_proj.weight"

    def vision_k_biases(self, layer_idx):
        """视觉编码器K偏置名"""
        return f"vision_tower.vision_model.encoder.layers.{layer_idx}.self_attn.k_proj.bias"

    def vision_v_weights(self, layer_idx):
        """视觉编码器V权重名"""
        return f"vision_tower.vision_model.encoder.layers.{layer_idx}.self_attn.v_proj.weight"

    def vision_v_biases(self, layer_idx):
        """视觉编码器V偏置名"""
        return f"vision_tower.vision_model.encoder.layers.{layer_idx}.self_attn.v_proj.bias"

    # def vision_qkv_weight(self, layer_idx):
    #     """视觉编码器QKV合并权重名（如果存在）"""
    #     # 某些实现可能将QKV合并为一个权重
    #     return f"vision_tower.vision_model.encoder.layers.{layer_idx}.self_attn.qkv.weight"

    # def vision_qkv_bias(self, layer_idx):
    #     """视觉编码器QKV合并偏置名（如果存在）"""
    #     return f"vision_tower.vision_model.encoder.layers.{layer_idx}.self_attn.qkv.bias"

    def vision_proj_weight(self, layer_idx):
        """视觉编码器投影权重名"""
        return f"vision_tower.vision_model.encoder.layers.{layer_idx}.self_attn.out_proj.weight"

    def vision_proj_bias(self, layer_idx):
        """视觉编码器投影偏置名"""
        return f"vision_tower.vision_model.encoder.layers.{layer_idx}.self_attn.out_proj.bias"

    def vision_in_layer_post_norm_weight(self, layer_idx):
        """视觉编码器后置归一化权重名"""
        return f"vision_tower.vision_model.encoder.layers.{layer_idx}.layer_norm2.weight"

    def vision_post_norm_bias(self, layer_idx):
        """视觉编码器后置归一化偏置名"""
        return f"vision_tower.vision_model.encoder.layers.{layer_idx}.layer_norm2.bias"

    def vision_mlp_fc1_weight(self, layer_idx):
        """视觉编码器MLP第一层权重名"""
        return f"vision_tower.vision_model.encoder.layers.{layer_idx}.mlp.fc1.weight"

    def vision_mlp_fc1_bias(self, layer_idx):
        """视觉编码器MLP第一层偏置名"""
        return f"vision_tower.vision_model.encoder.layers.{layer_idx}.mlp.fc1.bias"

    def vision_mlp_fc2_weight(self, layer_idx):
        """视觉编码器MLP第二层权重名"""
        return f"vision_tower.vision_model.encoder.layers.{layer_idx}.mlp.fc2.weight"

    def vision_mlp_fc2_bias(self, layer_idx):
        """视觉编码器MLP第二层偏置名"""
        return f"vision_tower.vision_model.encoder.layers.{layer_idx}.mlp.fc2.bias"






    def vision_post_norm_final_weight(self):
        """视觉编码器最终归一化权重名"""
        return "vision_tower.vision_model.post_layernorm.weight"

    def vision_post_norm_final_bias(self):
        """视觉编码器最终归一化偏置名"""
        return "vision_tower.vision_model.post_layernorm.bias"

    def projector_weight(self):
        """多模态投影器权重名"""
        return "multi_modal_projector.linear_1.weight"

    def projector_bias(self):
        """多模态投影器偏置名"""
        return "multi_modal_projector.linear_1.bias"

    def projector_weight_2(self):
        """多模态投影器第二层权重名"""
        return "multi_modal_projector.linear_2.weight"

    def projector_bias_2(self):
        """多模态投影器第二层偏置名"""
        return "multi_modal_projector.linear_2.bias"

    def attn_norm(self, layer_idx):
        """注意力归一化权重名"""
        return f"language_model.model.layers.{layer_idx}.input_layernorm.weight"

    def attn_q(self, layer_idx):
        """注意力Q权重名"""
        return f"language_model.model.layers.{layer_idx}.self_attn.q_proj.weight"

    def attn_k(self, layer_idx):
        """注意力K权重名"""
        return f"language_model.model.layers.{layer_idx}.self_attn.k_proj.weight"

    def attn_v(self, layer_idx):
        """注意力V权重名"""
        return f"language_model.model.layers.{layer_idx}.self_attn.v_proj.weight"

    def attn_o(self, layer_idx):
        """注意力O权重名"""
        return f"language_model.model.layers.{layer_idx}.self_attn.o_proj.weight"

    def attn_qkv(self, layer_idx):
        """注意力QKV合并权重名"""
        # 对于LLaMA，通常Q、K、V是分开的，但某些实现可能合并
        return f"language_model.model.layers.{layer_idx}.self_attn.qkv.weight"  # 可能不存在

    def attn_qkv_b(self, layer_idx):
        """注意力QKV合并偏置名"""
        return f"language_model.model.layers.{layer_idx}.self_attn.qkv.bias"  # 可能不存在

    def attn_q_norm(self, layer_idx):
        """注意力Q归一化权重名（用于某些优化）"""
        return f"language_model.model.layers.{layer_idx}.self_attn.q_norm.weight"  # 可能不存在

    def attn_k_norm(self, layer_idx):
        """注意力K归一化权重名（用于某些优化）"""
        return f"language_model.model.layers.{layer_idx}.self_attn.k_norm.weight"  # 可能不存在

    def ffn_gate_up(self, layer_idx):
        """FFN gate和up合并权重名（某些实现的优化）"""
        return f"language_model.model.layers.{layer_idx}.mlp.gate_up_proj.weight"  # 可能不存在

    def ffn_norm(self, layer_idx):
        """FFN归一化权重名"""
        return f"language_model.model.layers.{layer_idx}.post_attention_layernorm.weight"

    def ffn_gate(self, layer_idx):
        """FFN门控权重名"""
        return f"language_model.model.layers.{layer_idx}.mlp.gate_proj.weight"

    def ffn_up(self, layer_idx):
        """FFN上投影权重名"""
        return f"language_model.model.layers.{layer_idx}.mlp.up_proj.weight"

    def ffn_down(self, layer_idx):
        """FFN下投影权重名"""
        return f"language_model.model.layers.{layer_idx}.mlp.down_proj.weight"

class LlavaMetaFromLlava(LlavaMetaCStruct):
    def __init__(self, config, dtype=torch.float16, max_tokens=None):
        # Data type conversion
        if dtype == torch.float16:
            dt_ = DataType.INFINI_DTYPE_F16
        elif dtype == torch.float32:
            dt_ = DataType.INFINI_DTYPE_F32
        elif dtype == torch.bfloat16:
            dt_ = DataType.INFINI_DTYPE_BF16
        else:
            dt_ = DataType.INFINI_DTYPE_F16

        # Vision encoder meta (from vision_config)
        vision_config = config.get("vision_config", {})
        vision_meta = LlavaVisionMetaCStruct(
            image_size=vision_config.get("image_size", 336),
            patch_size=vision_config.get("patch_size", 14),
            num_patches=(vision_config.get("image_size", 336) // vision_config.get("patch_size", 14)) ** 2,
            vision_embed_dim=vision_config.get("hidden_size", 1024),
            vision_num_layers=vision_config.get("num_hidden_layers", 24),
            vision_num_heads=vision_config.get("num_attention_heads", 16),
            vision_intermediate_size=vision_config.get("intermediate_size", 4096),
            vision_epsilon=1e-5,  # 来自 transformers
        )

        # Language model meta (from text_config or main config)
        text_config = config.get("text_config", config)

        # Vicuna-7B-v1.5的完整配置 (LLaVA text_config可能不完整)
        vicuna_config = {
            "num_hidden_layers": 32,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "intermediate_size": 11008,
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-05,
            "vocab_size": 32000,
            "rope_theta": 10000.0,
            "head_dim": 128,  # 4096 // 32
        }

        # 合并配置：优先使用LLaVA的text_config，缺失的用Vicuna默认值
        language_meta = LlavaLanguageMetaCStruct(
            dt_logits=dt_,
            nlayer=text_config.get("num_hidden_layers", vicuna_config["num_hidden_layers"]),
            d=text_config.get("hidden_size", vicuna_config["hidden_size"]),
            nh=text_config.get("num_attention_heads", vicuna_config["num_attention_heads"]),
            nkvh=text_config.get("num_key_value_heads", vicuna_config["num_key_value_heads"]),
            dh=text_config.get("head_dim", vicuna_config["head_dim"]),
            di=text_config.get("intermediate_size", vicuna_config["intermediate_size"]),
            dctx=(
                text_config.get("max_position_embeddings", vicuna_config["max_position_embeddings"])
                if max_tokens is None else max_tokens
            ),
            dvoc=text_config.get("vocab_size", vicuna_config["vocab_size"]),
            epsilon=text_config.get("rms_norm_eps", vicuna_config["rms_norm_eps"]),
            theta=text_config.get("rope_theta", vicuna_config["rope_theta"]),
            end_token=2,
        )

        # Projector meta
        projector_meta = LlavaProjectorMetaCStruct(
            vision_embed_dim=vision_config.get("hidden_size", 1024),
            text_embed_dim=text_config.get("hidden_size", vicuna_config["hidden_size"]),
            projector_hidden_size=config.get("mm_hidden_size", 4096),
        )

        # Call parent constructor with three meta structures
        super().__init__(
            vision_meta=vision_meta,
            language_meta=language_meta,
            projector_meta=projector_meta,
        )
        self.torch_dtype_logits = dtype



class LlavaWeightsImpl(LlavaWeightsCStruct):
    def __init__(
        self,
        meta,
        naming,
        state_dict,
        torch_dt_mat=torch.float16,
        torch_dt_norm=torch.float16,
        ndev=1,
    ):
        nlayer = meta.language_meta.nlayer
        vision_nlayer = meta.vision_meta.vision_num_layers
        d = meta.language_meta.d
        di = meta.language_meta.di
        nh = meta.language_meta.nh
        nkvh = meta.language_meta.nkvh
        dh = meta.language_meta.dh

        # 数据类型转换
        if torch_dt_mat == torch.float16:
            self.dt_mat = DataType.INFINI_DTYPE_F16
        elif torch_dt_mat == torch.float32:
            self.dt_mat = DataType.INFINI_DTYPE_F32
        elif torch_dt_mat == torch.bfloat16:
            self.dt_mat = DataType.INFINI_DTYPE_BF16
        else:
            self.dt_mat = DataType.INFINI_DTYPE_F16

        if torch_dt_norm == torch.float16:
            self.dt_norm = DataType.INFINI_DTYPE_F16
        elif torch_dt_norm == torch.float32:
            self.dt_norm = DataType.INFINI_DTYPE_F32
        elif torch_dt_norm == torch.bfloat16:
            self.dt_norm = DataType.INFINI_DTYPE_BF16
        else:
            self.dt_norm = DataType.INFINI_DTYPE_F32

        # self.transpose_linear_weights = 1 if transpose_weight else 0
        self.nlayer = nlayer
        self.vision_nlayer = vision_nlayer

        # === 视觉编码器权重 ===
        # Patch嵌入权重
        if naming.vision_patch_embed_weight() in state_dict:
            self.vision_patch_embed_tensor = state_dict[naming.vision_patch_embed_weight()].to(torch_dt_mat)
            # print(f"[Python LlavaWeightsImpl] torch_dt_mat: {torch_dt_mat} ")  # torch.float16 
            # print(f"[Python LlavaWeightsImpl] vision_patch_embed_tensor shape: {self.vision_patch_embed_tensor.shape} ")
            self.vision_patch_embed_weight = self.vision_patch_embed_tensor.data_ptr()
            # print(f"[Python LlavaWeightsImpl] vision_patch_embed_weight pointer: {hex(self.vision_patch_embed_weight)} " )
            # print(f"[Python LlavaWeightsImpl] first 10 vision_patch_embed_weight: {self.vision_patch_embed_tensor.flatten()[:10]} ")
            # Print pointer address in 0x... format
            try:
                addr = int(self.vision_patch_embed_weight)
                # print(f"[Python LlavaWeightsImpl] vision_patch_embed_weight address: {hex(addr)}")
            except Exception as e:
                print(f"[Python LlavaWeightsImpl] failed to get vision_patch_embed_weight address: {e}")
        else:
            self.vision_patch_embed_weight = 0

        # 位置嵌入和class token
        if naming.vision_position_embedding() in state_dict:
            self.vision_position_embedding_tensor = state_dict[naming.vision_position_embedding()].to(torch_dt_mat)
            self.vision_position_embedding = self.vision_position_embedding_tensor.data_ptr()
        else:
            self.vision_position_embedding = 0

        # if naming.vision_class_embedding() in state_dict:
        #     self.vision_class_embedding_tensor = state_dict[naming.vision_class_embedding()].to(torch_dt_mat)
        #     self.vision_class_embedding = self.vision_class_embedding_tensor.data_ptr()
        if naming.vision_class_token() in state_dict:
            self.vision_class_token_tensor = state_dict[naming.vision_class_token()].to(torch_dt_mat)
            # print(f"[Python LlavaWeightsImpl] vision_class_token_tensor: {self.vision_class_token_tensor} ")
            # print(f"[Python LlavaWeightsImpl] vision_class_token_tensor shape: {self.vision_class_token_tensor.shape} " )
            # print(f"[Python LlavaWeightsImpl] vision_class_token_tensor dtype: {self.vision_class_token_tensor.dtype} " )
            self.vision_class_token = self.vision_class_token_tensor.data_ptr()
            print(f"[Python LlavaWeightsImpl] vision_class_token pointer: {hex(self.vision_class_token)} ")
        else:
            self.vision_class_token = 0

        # pre_layernorm.weight
        if naming.vision_pre_layernorm_weight() in state_dict:
            self.vision_pre_layernorm_weight_tensor = state_dict[naming.vision_pre_layernorm_weight()].to(torch_dt_mat)
            self.vision_pre_layernorm_weight = self.vision_pre_layernorm_weight_tensor.data_ptr()
            print(f"[Python LlavaWeightsImpl] vision_pre_layernorm_weight pointer: {hex(self.vision_pre_layernorm_weight)} ")
        else:
            self.vision_pre_layernorm_weight = 0

        # pre_layernorm.bias
        if naming.vision_pre_layernorm_bias() in state_dict:
            self.vision_pre_layernorm_bias_tensor = state_dict[naming.vision_pre_layernorm_bias()].to(torch_dt_mat)
            self.vision_pre_layernorm_bias = self.vision_pre_layernorm_bias_tensor.data_ptr()
        else:
            self.vision_pre_layernorm_bias = 0
        # post_layernorm.weight
        if naming.vision_post_layernorm_weight() in state_dict:
            self.vision_post_layernorm_weight_tensor = state_dict[naming.vision_post_layernorm_weight()].to(torch_dt_mat)
            self.vision_post_layernorm_weight = self.vision_post_layernorm_weight_tensor.data_ptr()
        else:
            self.vision_post_layernorm_weight = 0

        # post_layernorm.bias
        if naming.vision_post_layernorm_bias() in state_dict:
            self.vision_post_layernorm_bias_tensor = state_dict[naming.vision_post_layernorm_bias()].to(torch_dt_mat)
            self.vision_post_layernorm_bias = self.vision_post_layernorm_bias_tensor.data_ptr()
        else:
            self.vision_post_layernorm_bias = 0

        # in_layer pre_norm weights
        self.vision_in_layer_pre_norm_weight_tensors = [
            state_dict[naming.vision_in_layer_pre_norm_weights(i)].to(torch_dt_mat) for i in range(vision_nlayer)
        ]
        self.vision_in_layer_pre_norm_weight_ptrs = [
            self.vision_in_layer_pre_norm_weight_tensors[i].data_ptr() for i in range(vision_nlayer)
        ]
        self.vision_in_layer_pre_norm_weights = (c_void_p * vision_nlayer)(*self.vision_in_layer_pre_norm_weight_ptrs)

        # in_layer pre_norm biases
        self.vision_in_layer_pre_norm_bias_tensors = [
            state_dict[naming.vision_in_layer_pre_norm_biases(i)].to(torch_dt_mat) for i in range(vision_nlayer)
        ]
        self.vision_in_layer_pre_norm_bias_ptrs = [
            self.vision_in_layer_pre_norm_bias_tensors[i].data_ptr() for i in range(vision_nlayer)
        ]
        self.vision_in_layer_pre_norm_biases = (c_void_p * vision_nlayer)(*self.vision_in_layer_pre_norm_bias_ptrs)

        # q weights
        self.vision_q_weight_tensors = [
            state_dict[naming.vision_q_weights(i)].to(torch_dt_mat) for i in range(vision_nlayer)
        ]
        self.vision_q_weight_ptrs = [
            self.vision_q_weight_tensors[i].data_ptr() for i in range(vision_nlayer)
        ]
        self.vision_q_weights = (c_void_p * vision_nlayer)(*self.vision_q_weight_ptrs)
        # q biases
        self.vision_q_bias_tensors = [
            state_dict[naming.vision_q_biases(i)].to(torch_dt_mat) for i in range(vision_nlayer)
        ]
        self.vision_q_bias_ptrs = [
            self.vision_q_bias_tensors[i].data_ptr() for i in range(vision_nlayer)
        ]
        self.vision_q_biases = (c_void_p * vision_nlayer)(*self.vision_q_bias_ptrs)
        # k weights
        self.vision_k_weight_tensors = [
            state_dict[naming.vision_k_weights(i)].to(torch_dt_mat) for i in range(vision_nlayer)
        ]
        self.vision_k_weight_ptrs = [
            self.vision_k_weight_tensors[i].data_ptr() for i in range(vision_nlayer)
        ]
        self.vision_k_weights = (c_void_p * vision_nlayer)(*self.vision_k_weight_ptrs)
        # k biases
        self.vision_k_bias_tensors = [
            state_dict[naming.vision_k_biases(i)].to(torch_dt_mat) for i in range(vision_nlayer)
        ]
        self.vision_k_bias_ptrs = [
            self.vision_k_bias_tensors[i].data_ptr() for i in range(vision_nlayer)
        ]
        self.vision_k_biases = (c_void_p * vision_nlayer)(*self.vision_k_bias_ptrs)
        # v weights
        self.vision_v_weight_tensors = [
            state_dict[naming.vision_v_weights(i)].to(torch_dt_mat) for i in range(vision_nlayer)
        ]
        self.vision_v_weight_ptrs = [
            self.vision_v_weight_tensors[i].data_ptr() for i in range(vision_nlayer)
        ]
        self.vision_v_weights = (c_void_p * vision_nlayer)(*self.vision_v_weight_ptrs)
        # v biases
        self.vision_v_bias_tensors = [
            state_dict[naming.vision_v_biases(i)].to(torch_dt_mat) for i in range(vision_nlayer)
        ]
        self.vision_v_bias_ptrs = [
            self.vision_v_bias_tensors[i].data_ptr() for i in range(vision_nlayer)
        ]
        self.vision_v_biases = (c_void_p * vision_nlayer)(*self.vision_v_bias_ptrs)

        ###############################################
        # out_proj.weight / out_proj.bias
        ###############################################

        self.vision_proj_weight_tensors = [
            state_dict[naming.vision_proj_weight(i)].to(torch_dt_mat) for i in range(vision_nlayer)
        ]
        self.vision_proj_weight_ptrs = [
            self.vision_proj_weight_tensors[i].data_ptr() for i in range(vision_nlayer)
        ]
        self.vision_proj_weight = (c_void_p * vision_nlayer)(*self.vision_proj_weight_ptrs)

        self.vision_proj_bias_tensors = [
            state_dict[naming.vision_proj_bias(i)].to(torch_dt_mat) for i in range(vision_nlayer)
        ]
        self.vision_proj_bias_ptrs = [
            self.vision_proj_bias_tensors[i].data_ptr() for i in range(vision_nlayer)
        ]
        self.vision_proj_bias = (c_void_p * vision_nlayer)(*self.vision_proj_bias_ptrs)


        ###############################################
        # post norm (after attention) weight / bias
        ###############################################

        self.vision_in_layer_post_norm_tensors = [
            state_dict[naming.vision_in_layer_post_norm_weight(i)].to(torch_dt_mat) for i in range(vision_nlayer)
        ]
        self.vision_in_layer_post_norm_ptrs = [
            self.vision_in_layer_post_norm_tensors[i].data_ptr() for i in range(vision_nlayer)
        ]
        self.vision_in_layer_post_norm_weight = (c_void_p * vision_nlayer)(*self.vision_in_layer_post_norm_ptrs)
        # print(f"[Python LlavaWeightsImpl] vision_in_layer_post_norm_weight pointers: {[hex(ptr) for ptr in self.vision_in_layer_post_norm_ptrs]} ")

        self.vision_post_norm_bias_tensors = [
            state_dict[naming.vision_post_norm_bias(i)].to(torch_dt_mat) for i in range(vision_nlayer)
        ]
        self.vision_post_norm_bias_ptrs = [
            self.vision_post_norm_bias_tensors[i].data_ptr() for i in range(vision_nlayer)
        ]
        self.vision_post_norm_bias = (c_void_p * vision_nlayer)(*self.vision_post_norm_bias_ptrs)


        ###############################################
        # MLP: fc1 / fc2
        ###############################################

        # fc1.weight
        self.vision_mlp_fc1_weight_tensors = [
            state_dict[naming.vision_mlp_fc1_weight(i)].to(torch_dt_mat) for i in range(vision_nlayer)
        ]
        self.vision_mlp_fc1_weight_ptrs = [
            self.vision_mlp_fc1_weight_tensors[i].data_ptr() for i in range(vision_nlayer)
        ]
        self.vision_mlp_fc1_weight = (c_void_p * vision_nlayer)(*self.vision_mlp_fc1_weight_ptrs)

        # fc1.bias
        self.vision_mlp_fc1_bias_tensors = [
            state_dict[naming.vision_mlp_fc1_bias(i)].to(torch_dt_mat) for i in range(vision_nlayer)
        ]
        self.vision_mlp_fc1_bias_ptrs = [
            self.vision_mlp_fc1_bias_tensors[i].data_ptr() for i in range(vision_nlayer)
        ]
        self.vision_mlp_fc1_bias = (c_void_p * vision_nlayer)(*self.vision_mlp_fc1_bias_ptrs)


        # fc2.weight
        self.vision_mlp_fc2_weight_tensors = [
            state_dict[naming.vision_mlp_fc2_weight(i)].to(torch_dt_mat) for i in range(vision_nlayer)
        ]
        self.vision_mlp_fc2_weight_ptrs = [
            self.vision_mlp_fc2_weight_tensors[i].data_ptr() for i in range(vision_nlayer)
        ]
        self.vision_mlp_fc2_weight = (c_void_p * vision_nlayer)(*self.vision_mlp_fc2_weight_ptrs)

        # fc2.bias
        self.vision_mlp_fc2_bias_tensors = [
            state_dict[naming.vision_mlp_fc2_bias(i)].to(torch_dt_mat) for i in range(vision_nlayer)
        ]
        self.vision_mlp_fc2_bias_ptrs = [
            self.vision_mlp_fc2_bias_tensors[i].data_ptr() for i in range(vision_nlayer)
        ]
        self.vision_mlp_fc2_bias = (c_void_p * vision_nlayer)(*self.vision_mlp_fc2_bias_ptrs)



        # === 多模态投影器权重 ===
        if naming.projector_weight() in state_dict:
            self.projector_weight_tensor = state_dict[naming.projector_weight()].to(torch_dt_mat)
            self.projector_weight = self.projector_weight_tensor.data_ptr()
        else:
            self.projector_weight = 0

        if naming.projector_bias() in state_dict:
            self.projector_bias_tensor = state_dict[naming.projector_bias()].to(torch_dt_mat)
            self.projector_bias = self.projector_bias_tensor.data_ptr()
        else:
            self.projector_bias = 0

        # === 语言模型权重 (按照Jiuge模式) ===
        # 输入输出嵌入
        self.input_embd_tensor = state_dict[naming.input_embd()].to(torch_dt_mat)
        self.input_embd = self.input_embd_tensor.data_ptr()

        self.output_norm_tensor = state_dict[naming.output_norm()].to(torch_dt_mat)
        self.output_norm = self.output_norm_tensor.data_ptr()

        self.output_embd_tensor = state_dict[naming.output_embd()].to(torch_dt_mat)
        self.output_embd = self.output_embd_tensor.data_ptr()

        # 注意力权重数组
        self.attn_norm_tensors = [
            state_dict[naming.attn_norm(i)].to(torch_dt_mat) for i in range(nlayer)
        ]
        self.attn_norm_ptrs = [
            self.attn_norm_tensors[i].data_ptr() for i in range(nlayer)
        ]
        self.attn_norm = (c_void_p * nlayer)(*self.attn_norm_ptrs)

        # # QKV权重 - 对于LLaVA，Q、K、V是分开的，但我们可以按Jiuge的方式合并处理
        # def qkv_slices(_i):
        #     _Q = (
        #         state_dict[naming.attn_q(_i)]
        #         .reshape([nh, 2, dh // 2, d])
        #         .transpose(1, 2)
        #     )
        #     _K = (
        #         state_dict[naming.attn_k(_i)]
        #         .reshape([nkvh, 2, dh // 2, d])
        #         .transpose(1, 2)
        #     )
        #     _V = state_dict[naming.attn_v(_i)].reshape([nkvh, dh // 2, 2, d])
        #     _result = []
        #     _nh = nh // ndev
        #     _nkvh = nkvh // ndev
        #     for _idev in range(ndev):
        #         _result.append(_Q[_idev * _nh : (_idev + 1) * _nh, :, :, :])
        #         _result.append(_K[_idev * _nkvh : (_idev + 1) * _nkvh, :, :, :])
        #         _result.append(_V[_idev * _nkvh : (_idev + 1) * _nkvh, :, :])
        #     return _result

        # self.qkv_tensor = [
        #     torch.concat(qkv_slices(i)).to(torch_dt_mat) for i in range(nlayer)
        # ]
        # if not transpose_weight:
        #     for i in range(nlayer):
        #         self.qkv_tensor[i] = (
        #             self.qkv_tensor[i]
        #             .reshape(ndev, (nh + 2 * nkvh) // ndev * dh, d)
        #             .transpose(1, 2)
        #             .contiguous()
        #         )
        # self.qkv_tensor_ptrs = [self.qkv_tensor[i].data_ptr() for i in range(nlayer)]
        # self.attn_qkv = (c_void_p * nlayer)(*self.qkv_tensor_ptrs)

        # # QKV bias (LLaVA通常没有bias)
        # self.attn_qkv_b = (c_void_p * nlayer)()
        # for i in range(nlayer):
        #     self.attn_qkv_b[i] = 0

        # # Q norm 和 K norm (LLaVA通常没有)
        # self.attn_q_norm = (c_void_p * nlayer)()
        # self.attn_k_norm = (c_void_p * nlayer)()
        # for i in range(nlayer):
        #     self.attn_q_norm[i] = 0
        #     self.attn_k_norm[i] = 0

        # # Attention O权重
        # self.attn_o_tensor = [
        #     (
        #         state_dict[naming.attn_o(i)]
        #         .to(torch_dt_mat)
        #         .reshape([d, ndev, nh // ndev * dh])
        #         .transpose(0, 1)
        #         .contiguous()
        #         if transpose_weight
        #         else state_dict[naming.attn_o(i)]
        #         .transpose(0, 1)
        #         .to(torch_dt_mat)
        #         .contiguous()
        #     )
        #     for i in range(nlayer)
        # ]
        # self.attn_o_ptrs = [self.attn_o_tensor[i].data_ptr() for i in range(nlayer)]
        # self.attn_o = (c_void_p * nlayer)(*self.attn_o_ptrs)

        # # FFN权重
        # self.ffn_norm_tensors = [
        #     state_dict[naming.ffn_norm(i)].to(torch_dt_norm) for i in range(nlayer)
        # ]
        # self.ffn_norm_ptrs = [
        #     self.ffn_norm_tensors[i].data_ptr() for i in range(nlayer)
        # ]
        # self.ffn_norm = (c_void_p * nlayer)(*self.ffn_norm_ptrs)

        # def gate_up_slices(_i):
        #     _result = []
        #     _di = di // ndev
        #     for _idev in range(ndev):
        #         _start = _idev * _di
        #         _end = (_idev + 1) * _di
        #         _result.append(state_dict[naming.ffn_gate(_i)][_start:_end, :])
        #         _result.append(state_dict[naming.ffn_up(_i)][_start:_end, :])
        #     return _result

        # self.gate_up_tensors = [
        #     torch.concat(gate_up_slices(i)).to(torch_dt_mat) for i in range(nlayer)
        # ]
        # if not transpose_weight:
        #     for i in range(nlayer):
        #         self.gate_up_tensors[i] = (
        #             self.gate_up_tensors[i]
        #             .reshape(ndev, 2 * di // ndev, d)
        #             .transpose(1, 2)
        #             .contiguous()
        #         )
        # self.gate_up_ptrs = [self.gate_up_tensors[i].data_ptr() for i in range(nlayer)]
        # self.ffn_gate_up = (c_void_p * nlayer)(*self.gate_up_ptrs)

        # self.ffn_down_tensor = [
        #     (
        #         state_dict[naming.ffn_down(i)]
        #         .to(torch_dt_mat)
        #         .reshape([d, ndev, di // ndev])
        #         .transpose(0, 1)
        #         .contiguous()
        #         if transpose_weight
        #         else state_dict[naming.ffn_down(i)]
        #         .transpose(0, 1)
        #         .to(torch_dt_mat)
        #         .contiguous()
        #     )
        #     for i in range(nlayer)
        # ]
        # self.ffn_down_ptrs = [self.ffn_down_tensor[i].data_ptr() for i in range(nlayer)]
        # self.ffn_down = (c_void_p * nlayer)(*self.ffn_down_ptrs)

        # # === 视觉编码器权重数组 ===
        # vision_layer_size = meta.vision_meta.vision_num_layers
        # self.vision_encoder_weights = (c_void_p * (vision_layer_size * 10))()

        # # 填充视觉编码器权重 (简化版，实际应该按Jiuge模式处理)
        # for i in range(vision_layer_size):
        #     idx = i * 10
        #     # 这里简化处理，实际应该像Jiuge那样创建tensor对象并保存
        #     vision_pre_norm_key = naming.vision_pre_norm(i)
        #     if vision_pre_norm_key in state_dict:
        #         self.vision_encoder_weights[idx] = state_dict[vision_pre_norm_key].data_ptr()
        #     else:
        #         self.vision_encoder_weights[idx] = 0

        #     # 其他视觉权重类似处理...
        #     for j in range(1, 10):
        #         self.vision_encoder_weights[idx + j] = 0

        # 初始化父类结构
        super().__init__()



class LLaVAForCauslLM:
    def __init__(self, model_dir_path, device=DeviceType.DEVICE_TYPE_CPU, ndev=1, max_tokens=None):
        def load_all_safetensors_from_dir(dir_path_: str):
            tensors_ = {}
            dir_path_ = Path(dir_path_)
            for file in sorted(dir_path_.glob("*.safetensors")):
                data_ = safetensors.safe_open(file, "pt")
                for name_ in data_.keys():
                    tensors_[name_] = data_.get_tensor(name_)
            return tensors_


        # 内部三个组件
        self.preprocessor = AutoProcessor.from_pretrained(model_dir_path)
        # self.vision_encoder = LLaVAVisionEncoder(model_dir_path, device_type, ndev)
        # self.mm_projector = LLaVAMultiModalProjector(model_dir_path, device_type, ndev)
        # self.language_model = JiugeForCauslLM(model_dir_path, device_type, ndev)  # ✅ 复用
        print("Loading model weights to host...")
        load_start_time = time.time()

        with open(os.path.join(model_dir_path, "config.json"), "r") as f:
            config = json.load(f)
            self.config = config
        self.eos_token_id = [2]
        print(f"Model config: {self.config}")
        print(f"Model eos_token_id: {self.eos_token_id}")

        # transpose_weight = (
        #     device != DeviceType.DEVICE_TYPE_ASCEND
        # )  # y = xW is faster than y=xW^T on Ascend

        # print(f"device: {device}")
        self.llava_model = LlavaModel()

        if "llava" == config["model_type"]:
            print("Loading LLaVA model...")
            state_dict = load_all_safetensors_from_dir(model_dir_path)
            print(f"state_dict keys: {list(state_dict.keys())[:10]} ...")
            self.meta = LlavaMetaFromLlava(config, max_tokens=max_tokens)
            # print(f"meta type: {type(self.meta)}") # meta type: <class '__main__.LlavaMetaFromLlava'>
            # print(f"meta value: {self.meta}") # meta value: <__main__.LlavaMetaFromLlava object at 0x7fda3c5e91c0>
            self.weights = LlavaWeightsImpl(
                self.meta,
                LlavaWeightsNaming(),
                state_dict,
                ndev=ndev,
            )

            transpose_weight = (
                device != DeviceType.DEVICE_TYPE_ASCEND
            )  # y = xW is faster than y=xW^T on Ascend


            self.language_meta = JiugeMetaFromLlama(config, max_tokens=max_tokens)
            self.language_weights = JiugeWeightsImpl(
                self.language_meta,
                LlamaWeightsNaming(),
                state_dict,
                ndev=ndev,
                transpose_weight=transpose_weight,
            )
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_dir_path
            )

            # print(f"weights type: {type(self.weights)}") # weights type: <class '__main__.LlavaWeightsImpl'>
            # print(f"weights value: {self.weights}") # weights value: <__main__.LlavaWeightsImpl object at 0x7fda3c5e9340>
        load_end_time = time.time()
        print(f"Time used: {load_end_time - load_start_time:.3f}s")
        print(f"Creating model on {ndev} devices...")
        self.dev_ids = (c_int * ndev)(*[i for i in range(ndev)])
        self.ndev = ndev
        self.device = device

        self.model_instance = self.llava_model.create_model(
            byref(self.meta),
            byref(self.weights),
            device,
            ndev,
            self.dev_ids,
        )

        load_end_time = time.time()
        print(f"Time used: {load_end_time - load_start_time:.3f}s")


    def max_context_len(self):
        return self.meta.language_meta.dctx

    def create_kv_cache(self):
        """创建 LLaVA 的 KV Cache"""
        # 调用 C++ 层的 createKVCache 函数
        # 参数：nlayer, max_len, nkvh, dk, dv, dtype, device, dev_ids, ndev
        return self.llava_model.create_kv_cache(
            self.meta.language_meta.nlayer,        # 语言模型层数
            self.meta.language_meta.dctx,           # 最大上下文长度
            self.meta.language_meta.nkvh,           # key-value head 数
            self.meta.language_meta.dh,            # head 维度
            self.meta.language_meta.dh,            # value 维度 (通常与dh相同)
            self.meta.language_meta.dt_logits,      # 数据类型
            self.device,                             # 设备类型
            self.dev_ids,                            # 设备ID列表
            self.ndev                               # 设备数量
        )
    
    def debug_image(self, ptr, pixel_values, num):
        print("tensor数组:", pixel_values.flatten()[:num].tolist())
        num_values = pixel_values.numel()
        # 数值数组
        values_list = []
        # 二进制表示（uint16或二进制字符串）
        binary_list = []

        for i in range(num_values):
            addr = ptr + i * 2
            raw_uint16 = ctypes.c_uint16.from_address(addr).value
            
            # 正确解读位模式为float16
            float16_val = np.array([raw_uint16], dtype=np.uint16).view(np.float16)[0]
            
            values_list.append(float(float16_val))  # 转为Python float
            binary_list.append(f"{raw_uint16:016b}")

        print("数值数组:", values_list[:num])
        print("二进制数组:", binary_list[:num])


    def drop_kv_cache(self, kv_cache):
        """删除 LLaVA 的 KV Cache"""
        self.llava_model.drop_kv_cache(kv_cache)

    # === LLaVA四阶段推理方法 ===
    def batch_infer_encode(self, pixel_values, input_tokens_list):
        """阶段1: Vision Encoder - 将图像编码为视觉特征"""
        if pixel_values is None:
            return None
        # print(f"pixels value:{pixel_values.flatten()[:2000].tolist()}")

        # 将torch tensor转换为连续的字节数据
        if hasattr(pixel_values, 'contiguous'):
            pixel_values = pixel_values.contiguous()

        # 获取图像数据指针
        image_data_fp16 = pixel_values.to(torch.float16).cpu()
        image_data = image_data_fp16.data_ptr()
        # self.debug_image(image_data, image_data_fp16, 100)
        print(f"pixel_values shape: {pixel_values.shape}")
        print(f"image_data pointer: {hex(image_data)}")
        batch_size = pixel_values.shape[0] if len(pixel_values.shape) > 0 else 1

        # 准备输出缓冲区（视觉特征）
        vision_features_size = self.meta.vision_meta.vision_embed_dim * self.meta.vision_meta.num_patches
        vision_features_output = torch.zeros(vision_features_size, dtype=torch.float16)

        print(f"Vision encoding: input shape {pixel_values.shape} -> features size {vision_features_size}")

        # 准备输出缓冲区
        output_buffer = vision_features_output.data_ptr()


        # 调用C++层的 infer_batch_vision 函数
        self.llava_model.infer_batch_vision(
            self.model_instance,
            image_data,
            output_buffer
        )
        print(f"output_buffer pointer: {hex(output_buffer)}")
        # print(f"output_buffer: {list(output_buffer)}")
        # # # # # # # # # # # # # # # # # # 
        # try:
        #     flat = pixel_values.detach().cpu().flatten()
        #     first10 = flat[:10].tolist()
        #     formatted = ", ".join(f"{float(x):.6f}" for x in first10)
        #     print(f"pixel_values first 10: {formatted}")
        # except Exception:
        #     # Fallback if not a tensor or other error
        #     print(f"pixel_values: {pixel_values}")
        # # # # # # # # # # # # # # # # # # 

        return vision_features_output


    def batch_infer_compressor(self, features, kv_caches):
        """阶段4: KV-Cache Compression - 压缩KV缓存以节省内存"""
        if kv_caches is None:
            print("=== KV-Cache Compression Skipped (No KV Caches) ===")
            return kv_caches

        print("=== LLaVA KV-Cache Compression ===")

        # TODO: 集成Fastcache的压缩算法
        print("KV-Cache compression: (Future - Fastcache integration)")

        return kv_caches


    def generate(
        self, 
        messages, 
        # max_steps, // jiuge有，我暂时没有
        topp_=1.0,
        topk_=1,
        temperature_=1.0,
        verbose=False):
        mm_inputs = self.preprocessor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        pixel_values = mm_inputs.pixel_values
        attention_mask = mm_inputs.attention_mask
        input_ids = mm_inputs.input_ids
        print(f"Input token IDs shape: {input_ids.shape}")

        # 将torch tensor转换为Python列表，就像jiuge.py那样
        if hasattr(input_ids, 'flatten'):
            input_ids_list = input_ids.flatten().tolist()
        else:
            input_ids_list = input_ids.tolist()

        infer_task = InferTask(
            0,
            input_ids_list,  # 使用列表而不是tensor
            self.max_context_len(),
            temperature_,
            topk_,
            topp_,
            end_tokens=self.eos_token_id,
        )
        print(f"KV Cache: {infer_task.kvcache()}")

        # # 更详细的调试信息
        # print("\n=== InferTask 详细信息 ===")
        # print(repr(infer_task))

        # infer_task.bind_kvcache(KVCache(self))

        # print("\n=== bind_kvcache 后 KV Cache 详细信息 ===")
        # print(repr(infer_task.kvcache()))

        steps = 0
        total_time = 0
        prefill_time = 0
        decode_time = 0
        output_content = ""

        # === LLaVA四阶段推理流程 ===
        # 阶段1: Vision Encoder - 将图像编码为视觉特征
        output_encode = self.batch_infer_encode(pixel_values, input_ids_list)
        print(f"Output encode shape: {output_encode.shape if output_encode is not None else 'None'}")
        print(f"Output encode: {output_encode if output_encode is not None else 'None'}")


        output_encode = self.batch_infer_llm(pixel_values, input_ids_list)



        # # 阶段2: MultiModal Projector - 将视觉特征投影到文本嵌入空间
        # projected_features = self.batch_infer_projector(output_encode, input_ids_list)

        # # 阶段3: Language Model - 处理文本tokens和投影的视觉特征
        # output_tokens = self.batch_infer_language([input_ids_list], [infer_task.kvcache()], projected_features)

        # # 阶段4: KV-Cache Compression - 压缩KV缓存 (可选)
        # compressed_kv_cache = self.batch_infer_compressor(projected_features, [infer_task.kvcache()])
        # steps += 1


        output_str = self.tokenizer.decode(output_tokens[0])
        print(f"Decoded output from prefill: {output_str}\n")
        output_content += output_str
        print(output_str, end="", flush=True)
        if output_tokens[0] in self.eos_token_id:

            total_tokens = len(tokens) + 1  # input tokens + first output token

            return output_content, total_time * 1000

        infer_task.next(output_tokens[0])
        for step_i in range(1, max_steps):
            output_tokens = self.batch_infer_one_round([infer_task])
            steps += 1
            output_str = self.tokenizer.decode(output_tokens[0])
            output_content += output_str
            if output_tokens[0] in self.eos_token_id:
                break
            infer_task.next(output_tokens[0])

        steps += 1

















        if verbose:
            print("LLaVAForConditionalGeneration.generate:")
            print(f"  pixel_values.shape: {pixel_values.shape}")
            print(f"  attention_mask.shape: {attention_mask.shape}")
            print(f"  input_ids.shape: {input_ids.shape}")
        # TODO: 2. 视觉编码
        # vision_features = self.vision_encoder.encode(image_tensor)

        # TODO: 3. 多模态投影
        # image_tokens = self.mm_projector.project(vision_features)

        # TODO: 4. Token融合
        # combined_tokens = self._fuse_tokens(prompt, image_tokens)

        # TODO: 5. 语言模型生成 (复用Jiuge)
        # return self.language_model.generate_tokens(combined_tokens, max_new_tokens, verbose)




def test():
    if len(sys.argv) < 3:
        print(
            "Usage: python jiuge.py [--cpu | --nvidia| --cambricon | --ascend | --metax | --moore | --iluvatar | --kunlun | --hygon] <path/to/model_dir> [n_device] [--verbose]"
        )
        sys.exit(1)

    # Parse command line arguments
    model_path = sys.argv[2]
    device_type = DeviceType.DEVICE_TYPE_CPU
    verbose = False

    # Check for verbose flag
    for arg in sys.argv:
        if arg == "--verbose":
            verbose = True
            break

    if sys.argv[1] == "--cpu":
        device_type = DeviceType.DEVICE_TYPE_CPU
    elif sys.argv[1] == "--nvidia":
        device_type = DeviceType.DEVICE_TYPE_NVIDIA
    elif sys.argv[1] == "--cambricon":
        device_type = DeviceType.DEVICE_TYPE_CAMBRICON
    elif sys.argv[1] == "--ascend":
        device_type = DeviceType.DEVICE_TYPE_ASCEND
    elif sys.argv[1] == "--metax":
        device_type = DeviceType.DEVICE_TYPE_METAX
    elif sys.argv[1] == "--moore":
        device_type = DeviceType.DEVICE_TYPE_MOORE
    elif sys.argv[1] == "--iluvatar":
        device_type = DeviceType.DEVICE_TYPE_ILUVATAR
    elif sys.argv[1] == "--kunlun":
        device_type = DeviceType.DEVICE_TYPE_KUNLUN
    elif sys.argv[1] == "--hygon":
        device_type = DeviceType.DEVICE_TYPE_HYGON
    else:
        print(
            "Usage: python jiuge.py [--cpu | --nvidia| --cambricon | --ascend | --metax | --moore | --iluvatar | --kunlun | --hygon] <path/to/model_dir> [n_device] [--verbose]"
        )
        sys.exit(1)

    # Find n_device argument (skip --verbose)
    ndev_args = [arg for arg in sys.argv[3:] if arg != "--verbose"]
    ndev = int(ndev_args[0]) if ndev_args else 1

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": "scripts/img/47_42.jpg"},
                {"type": "text", "text": "Describe this image."}
            ]
        },
    ]

    model = LLaVAForCauslLM(model_path, device_type, ndev)
    model.generate(messages, verbose=verbose)
    # model.destroy_model_instance()


if __name__ == "__main__":
    test()
