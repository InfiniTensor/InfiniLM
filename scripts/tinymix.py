from typing import List
from libinfinicore_infer import (
    TinyMixMetaCStruct,
    TinyMixWeightsCStruct,
    KVCacheCStruct,
    DataType,
    DeviceType,
    create_tinymix_model,
    destroy_tinymix_model,
    create_tinymix_kv_cache,
    drop_tinymix_kv_cache,
    infer_batch_tinymix,
)
from infer_task import InferTask, KVCache

from ctypes import POINTER, c_float, c_int, c_uint, c_void_p, byref
import os
from pathlib import Path
import safetensors
import sys
import time
import json
import math
import torch
import transformers

torch.set_default_device("cpu")


class TinyMixWeightsNaming:
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

    def ffn_norm(self, i):
        return f"model.layers.{i}.post_attention_layernorm.weight"

    def ffn_gate(self, i):
        return f"model.layers.{i}.block_sparse_moe.gate.weight"

    def ffn_gate_up(self, i, expert_idx):
        return f"model.layers.{i}.block_sparse_moe.experts.{expert_idx}.w1.weight"

    def ffn_down(self, i, expert_idx):
        return f"model.layers.{i}.block_sparse_moe.experts.{expert_idx}.w3.weight"
    
    def ffn_up(self, i, expert_idx):
        return f"model.layers.{i}.block_sparse_moe.experts.{expert_idx}.w2.weight"

    def match(state_dict):
        return "model.layers.0.block_sparse_moe.gate.weight" in state_dict


class TinyMixMetaFromConfig(TinyMixMetaCStruct):
    def __init__(self, config, dtype=torch.float16, max_tokens=None):
        dt_ = DataType.INFINI_DTYPE_F16
        if dtype == torch.float32:
            dt_ = DataType.INFINI_DTYPE_F32
        elif dtype == torch.bfloat16:
            dt_ = DataType.INFINI_DTYPE_BF16

        super().__init__(
            dt_logits=dt_,
            nlayer=config["num_hidden_layers"],
            d=config["hidden_size"],
            nh=config["num_attention_heads"],
            nkvh=config.get("num_key_value_heads", config["num_attention_heads"]),
            dh=config["hidden_size"] // config["num_attention_heads"],
            di=config["intermediate_size"],
            dctx=config.get("max_position_embeddings", max_tokens),
            dvoc=config["vocab_size"],
            nexpert=config.get("num_local_experts", 1),
            n_expert_activate=config.get("num_experts_per_tok", 1),
            epsilon=config["rms_norm_eps"],
            theta=config.get("rope_theta", 10000.0),
            end_token=config["eos_token_id"],
        )
        self.torch_dtype_logits = dtype


class TinyMixWeightsImpl(TinyMixWeightsCStruct):
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

        self.transpose_linear_weights = 1 if transpose_weight else 0
        self.nlayer = nlayer
        self.input_embd_tensor = (
            state_dict[naming.input_embd()].to(torch_dt_logits)
        )
        self.input_embd = self.input_embd_tensor.data_ptr()
        self.output_norm_tensor = (
            state_dict[naming.output_norm()].to(torch_dt_norm)
        )
        self.output_norm = self.output_norm_tensor.data_ptr()
        self.output_embd_tensor = state_dict[naming.output_embd()].to(torch_dt_mat)
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

        self.attn_qkv_b = None

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

        self.ffn_gate_tensors = [
            state_dict[naming.ffn_gate(i)].to(torch_dt_mat)
            for i in range(meta.nlayer)
        ]
        self.ffn_gate_ptrs = [t.data_ptr() for t in self.ffn_gate_tensors]
        self.ffn_gate = (c_void_p * meta.nlayer)(*self.ffn_gate_ptrs)

        self.ffn_gate_up_tensors = []
        self.ffn_down_tensors = []
        
        for i in range(meta.nlayer):
            gate_up_experts = []
            down_experts = []
            for j in range(meta.nexpert):
                gate = state_dict[naming.ffn_gate_up(i, j)].to(torch_dt_mat)
                up = state_dict[naming.ffn_up(i, j)].to(torch_dt_mat)
                gate_up_experts.append(torch.cat([gate, up], dim=0))
                down_experts.append(state_dict[naming.ffn_down(i, j)].to(torch_dt_mat))

            self.ffn_gate_up_tensors.append(gate_up_experts)
            self.ffn_down_tensors.append(down_experts)

        # Create nested pointers for C API
        self.ffn_gate_up_expert_ptrs = [
            (c_void_p * meta.nexpert)(*[t.data_ptr() for t in layer_tensors])
            for layer_tensors in self.ffn_gate_up_tensors
        ]
        self.ffn_gate_up = (POINTER(c_void_p) * meta.nlayer)(*self.ffn_gate_up_expert_ptrs)

        self.ffn_down_expert_ptrs = [
            (c_void_p * meta.nexpert)(*[t.data_ptr() for t in layer_tensors])
            for layer_tensors in self.ffn_down_tensors
        ]
        self.ffn_down = (POINTER(c_void_p) * meta.nlayer)(*self.ffn_down_expert_ptrs)

# Rest of the implementation will be similar to jiuge.py, but using the TinyMix classes
# and C functions. For brevity, this part is omitted but would include:
# - JiugeBatchedTask (can be reused)
# - A TinyMixForCauslLM class
# - A test() function to run inference from the command line

class TinyMixForCauslLM:
    def __init__(
        self, model_dir_path, device=DeviceType.DEVICE_TYPE_CPU, ndev=1, max_tokens=None
    ):
        # ... (loading logic similar to JiugeForCauslLM)
        
        with open(os.path.join(model_dir_path, "config.json"), "r") as f:
            config = json.load(f)
        
        state_dict = {}
        for file in sorted(Path(model_dir_path).glob("*.safetensors")):
            data = safetensors.safe_open(file, "pt")
            for name in data.keys():
                state_dict[name] = data.get_tensor(name)

        naming = TinyMixWeightsNaming()
        self.meta = TinyMixMetaFromConfig(config, max_tokens=max_tokens)
        self.weights = TinyMixWeightsImpl(self.meta, naming, state_dict, ndev=ndev)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir_path)

        dev_ids = (c_int * ndev)(*[i for i in range(ndev)])
        self.model_instance = create_tinymix_model(
            byref(self.meta),
            byref(self.weights),
            device,
            ndev,
            dev_ids,
        )

    # ... (generate, batch_infer_one_round etc. adapted for TinyMix)
    def batch_infer_one_round(self, tasks: List[InferTask]):
        output = (c_uint * len(tasks))()
        # JiugeBatchedTask can likely be reused directly
        batch_inputs = JiugeBatchedTask(tasks) 
        infer_batch_tinymix(
            self.model_instance,
            *(batch_inputs.input_args()),
            output,
        )
        return list(output)

    def destroy_model_instance(self):
        destroy_tinymix_model(self.model_instance)
        print("Model destroyed")
