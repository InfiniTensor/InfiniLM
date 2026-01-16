# 文件路径: icinfer/engine/weights_loader.py

import os
import json
import torch
import transformers
from typing import Tuple
import math
from ctypes import POINTER, c_float, c_int, c_uint, c_void_p, byref
import ctypes
import os
from pathlib import Path
import safetensors
import sys
import time
import json
import math
import torch
import transformers

from icinfer.engine.libinfinicore_infer import (
    JiugeMetaCStruct,
    JiugeWeightsCStruct,
    DataType,
    create_jiuge_model,
    DeviceType,
)
from icinfer.config import Config

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LlamaWeightsNaming:
    def input_embd(self):
        return "model.embed_tokens.weight"

    def output_norm(self):
        return "model.norm.weight"

    def output_embd(self):
        return "lm_head.weight"

    def attn_norm(self, i):
        return f"model.layers.{i}.input_layernorm.weight"

    def attn_q(self, i):
        return f"model.layers.{i}.self_attn.q_proj.weight"

    def attn_k(self, i):
        return f"model.layers.{i}.self_attn.k_proj.weight"

    def attn_v(self, i):
        return f"model.layers.{i}.self_attn.v_proj.weight"

    def attn_o(self, i):
        return f"model.layers.{i}.self_attn.o_proj.weight"

    def attn_q_b(self, i):
        return f"model.layers.{i}.self_attn.q_proj.bias"

    def attn_k_b(self, i):
        return f"model.layers.{i}.self_attn.k_proj.bias"

    def attn_v_b(self, i):
        return f"model.layers.{i}.self_attn.v_proj.bias"

    def attn_q_norm(self, i):
        return f"model.layers.{i}.self_attn.q_norm.weight"

    def attn_k_norm(self, i):
        return f"model.layers.{i}.self_attn.k_norm.weight"

    def ffn_norm(self, i):
        return f"model.layers.{i}.post_attention_layernorm.weight"

    def gate(self, i):
        return f"model.layers.{i}.mlp.gate_proj.weight"

    def up(self, i):
        return f"model.layers.{i}.mlp.up_proj.weight"

    def down(self, i):
        return f"model.layers.{i}.mlp.down_proj.weight"

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
            config.model_type in ["fm9g", "minicpm"]
            and hasattr(config, "scale_emb")
            and hasattr(config, "scale_depth")
            and hasattr(config, "dim_model_base")
        ):
            self.scale_input = config.scale_emb
            self.scale_output = config.hidden_size // config.dim_model_base
            self.scale_o = config.scale_depth / math.sqrt(config.num_hidden_layers)
            self.scale_down = config.scale_depth / math.sqrt(config.num_hidden_layers)

        dim_model_base = (
            config.dim_model_base if hasattr(config, "dim_model_base") else config.hidden_size
        )

        # Load longrope configuration
        rope_type = 0  # 0 = standard, 1 = longrope
        original_max_position_embeddings = 0
        short_factor_ptr = None
        long_factor_ptr = None
        self._short_factor_array = None  # Keep reference to prevent GC
        self._long_factor_array = None   # Keep reference to prevent GC

        # Handle both dict and object config
        if hasattr(config, "rope_scaling"):
            rope_scaling = config.rope_scaling
        elif isinstance(config, dict) and "rope_scaling" in config:
            rope_scaling = config["rope_scaling"]
        else:
            rope_scaling = {}

        if isinstance(rope_scaling, dict):
            rope_scaling_type = rope_scaling.get("rope_type") or rope_scaling.get("type", "")
            if rope_scaling_type == "longrope":
                rope_type = 1
                original_max_position_embeddings = rope_scaling.get(
                    "original_max_position_embeddings",
                    getattr(config, "original_max_position_embeddings", 0) if not isinstance(config, dict) else config.get("original_max_position_embeddings", 0)
                )

                short_factor_list = rope_scaling.get("short_factor", [])
                long_factor_list = rope_scaling.get("long_factor", [])

                if short_factor_list and long_factor_list:
                    # Convert to ctypes arrays
                    half_dh = (config.hidden_size // config.num_attention_heads) // 2
                    if len(short_factor_list) == half_dh and len(long_factor_list) == half_dh:
                        self._short_factor_array = (c_float * half_dh)(*short_factor_list)
                        self._long_factor_array = (c_float * half_dh)(*long_factor_list)
                        short_factor_ptr = ctypes.cast(self._short_factor_array, POINTER(c_float))
                        long_factor_ptr = ctypes.cast(self._long_factor_array, POINTER(c_float))
                    else:
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.warning(
                            f"Longrope factor arrays have wrong length: "
                            f"short={len(short_factor_list)}, long={len(long_factor_list)}, expected={half_dh}"
                        )

        super().__init__(
            dt_logits=dt_,
            nlayer=config.num_hidden_layers,
            d=config.hidden_size,
            nh=config.num_attention_heads,
            nkvh=(
                config.num_key_value_heads
                if hasattr(config, "num_key_value_heads")
                else config.num_attention_heads
            ),
            dh=config.hidden_size // config.num_attention_heads,
            di=config.intermediate_size,
            dctx=(config.max_position_embeddings if max_tokens is None else max_tokens),
            dvoc=config.vocab_size,
            kvcache_block_size=config.kvcache_block_size,
            dim_model_base=dim_model_base,
            epsilon=config.rms_norm_eps,
            theta=(config.rope_theta if hasattr(config, "rope_theta") else 100000.0),
            end_token=2,
            rope_type=rope_type,
            original_max_position_embeddings=original_max_position_embeddings,
            short_factor=short_factor_ptr,
            long_factor=long_factor_ptr,
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
            state_dict[naming.output_norm()].to(torch_dt_norm)
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


def load_weights_to_cpu(
    config: Config, device: DeviceType
) -> Tuple[JiugeMetaCStruct, JiugeWeightsImpl]:
    """
    复用旧 infiniinfer 的权重加载逻辑。
    在 CPU 上加载模型权重和配置，并将其转换为 C++ 兼容的结构体。
    """

    def load_all_safetensors_from_dir(dir_path_: str):
        tensors_ = {}
        dir_path_ = Path(dir_path_)
        for file in sorted(dir_path_.glob("*.safetensors")):
            data_ = safetensors.safe_open(file, "pt")
            for name_ in data_.keys():
                tensors_[name_] = data_.get_tensor(name_)
        return tensors_

    max_tokens = config.max_model_len
    model_dir_path = config.model_path
    ndev = config.tensor_parallel_size
    hf_config = config.hf_config

    print("Loading model weights to host...")
    load_start_time = time.time()

    transpose_weight = (
        device != DeviceType.DEVICE_TYPE_ASCEND
    )  # y = xW is faster than y=xW^T on Ascend
    if "llama" == hf_config.model_type:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_dir_path,
            device_map="cpu",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        load_statets_time = time.time()
        meta = JiugeMetaFromLlama(hf_config, max_tokens=max_tokens)
        weights = JiugeWeightsImpl(
            meta,
            LlamaWeightsNaming(),
            model.state_dict(),
            ndev=ndev,
            transpose_weight=transpose_weight,
        )
    elif "fm9g" == hf_config.model_type:
        logger.info(f"fm9g load start.")
        # )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_dir_path,
            device_map="cpu",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        logger.info(f"load over.")
        load_statets_time = time.time()
        meta = JiugeMetaFromLlama(hf_config, max_tokens=max_tokens)
        weights = JiugeWeightsImpl(
            meta,
            LlamaWeightsNaming(),
            model.state_dict(),
            ndev=ndev,
            transpose_weight=transpose_weight,
        )
    elif "fm9g7b" == hf_config.model_type:
        logger.info(f"fm9g7b load start.")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_dir_path,
            device_map="cpu",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        logger.info(f"load over.")
        load_statets_time = time.time()
        meta = JiugeMetaFromLlama(hf_config, max_tokens=max_tokens)
        weights = JiugeWeightsImpl(
            meta,
            LlamaWeightsNaming(),
            model.state_dict(),
            ndev=ndev,
            transpose_weight=transpose_weight,
        )
    elif "qwen2" == hf_config.model_type:
        state_dict = load_all_safetensors_from_dir(model_dir_path)
        if LlamaWeightsNaming.match(state_dict):
            meta = JiugeMetaFromLlama(hf_config, max_tokens=max_tokens)
            weights = JiugeWeightsImpl(
                meta,
                LlamaWeightsNaming(),
                state_dict,
                ndev=ndev,
                transpose_weight=transpose_weight,
            )
    else:
        raise ValueError("Unsupported model architecture")

    load_end_time = time.time()
    logger.info(
        f"Time overall used: {load_end_time - load_start_time:.3f}s, "
        f"load_states_time: {load_statets_time - load_start_time:.3f}s, "
        f"load_weights_impl_time: {load_end_time - load_statets_time:.3f}s"
    )

    logger.info(f"Creating model on {ndev} devices...")
    load_start_time = time.time()

    print("Weights loaded to CPU successfully.")
    return meta, weights


def load_model(config: Config, device: DeviceType):
    ndev = config.tensor_parallel_size
    meta, weights = load_weights_to_cpu(config, device)
    dev_ids = (c_int * ndev)(*[i for i in range(ndev)])
    model = create_jiuge_model(
        byref(meta),
        byref(weights),
        device,
        ndev,
        dev_ids,
    )
    return model, meta
