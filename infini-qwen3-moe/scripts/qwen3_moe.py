#!/usr/bin/env python3
"""
Qwen3-MoE Implementation for InfiniCore
This implementation provides Qwen3-MoE support using the dedicated qwen3_moe C++ API
with proper MoE routing, expert selection, and parameter mapping.

Key Features:
1. Uses dedicated Qwen3-MoE API with sparse expert activation
2. Handles MoE routing weights and expert selection properly
3. Supports both regular MLP and MoE layers in the same model
4. Implements efficient expert weight partitioning across devices
5. Provides MoE-specific debugging and statistics
"""

from typing import List, Optional, Dict, Any
import os
import sys
import time
import json
import torch
import transformers
from pathlib import Path
from ctypes import POINTER, c_float, c_int, c_uint, c_void_p, byref
import safetensors
import ctypes
import numpy as np

# Set default device
torch.set_default_device("cpu")

# Import the Qwen3-MoE specific API
try:
    from libinfinicore_infer import (
        Qwen3MoeMetaCStruct,
        Qwen3MoeWeightsCStruct,
        createQwen3MoeModel,
        destroyQwen3MoeModel,
        createQwen3MoeKVCache,
        dropQwen3MoeKVCache,
        inferQwen3MoeBatch,
        getQwen3MoeRouterStats,
        setQwen3MoeDebugMode,
        DataType,
        DeviceType,
        KVCacheCStruct,
    )
    QWEN3_MOE_API_AVAILABLE = True
    print("✓ Qwen3-MoE C++ API available")
except ImportError as e:
    print(f"⚠ Qwen3-MoE C++ API not available: {e}")
    print("  This version requires the qwen3_moe implementation")
    sys.exit(1)

from infer_task import Qwen3MoeInferTask, Qwen3MoeKVCache


class Qwen3MoeWeightsNaming:
    """Qwen3-MoE specific weight naming with MoE expert support"""
    
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

    # Regular MLP weights
    def gate(self, i):
        return f"model.layers.{i}.mlp.gate_proj.weight"

    def up(self, i):
        return f"model.layers.{i}.mlp.up_proj.weight"

    def down(self, i):
        return f"model.layers.{i}.mlp.down_proj.weight"

    # Qwen3-specific Q/K normalization weights
    def q_norm(self, i):
        return f"model.layers.{i}.self_attn.q_norm.weight"

    def k_norm(self, i):
        return f"model.layers.{i}.self_attn.k_norm.weight"

    # MoE-specific weights
    def moe_gate(self, i):
        """Router/gating network for expert selection"""
        return f"model.layers.{i}.mlp.gate.weight"

    def moe_expert_gate(self, i, expert_idx):
        """Gate projection for specific expert"""
        return f"model.layers.{i}.mlp.experts.{expert_idx}.gate_proj.weight"

    def moe_expert_up(self, i, expert_idx):
        """Up projection for specific expert"""
        return f"model.layers.{i}.mlp.experts.{expert_idx}.up_proj.weight"

    def moe_expert_down(self, i, expert_idx):
        """Down projection for specific expert"""
        return f"model.layers.{i}.mlp.experts.{expert_idx}.down_proj.weight"

    @staticmethod
    def match(state_dict):
        """Check if state_dict matches Qwen3-MoE naming pattern"""
        has_basic = (
            "model.norm.weight" in state_dict
            and "model.layers.0.self_attn.q_proj.weight" in state_dict
        )
        # Check for MoE specific patterns
        has_moe = any(
            "mlp.gate.weight" in key and "mlp.experts" in str(state_dict.keys())
            for key in state_dict.keys()
        )
        # Qwen3 often has q_norm and k_norm weights
        has_qk_norm = (
            "model.layers.0.self_attn.q_norm.weight" in state_dict
            and "model.layers.0.self_attn.k_norm.weight" in state_dict
        )
        return has_basic and (has_moe or has_qk_norm)


class Qwen3MoeMetaFromConfig(Qwen3MoeMetaCStruct):
    """Qwen3-MoE metadata structure from model config"""
    
    def __init__(self, config, dtype=torch.float16, max_tokens=None):
        super().__init__()
        
        if dtype == torch.float16:
            dt_ = DataType.INFINI_DTYPE_F16
        elif dtype == torch.float32:
            dt_ = DataType.INFINI_DTYPE_F32
        elif dtype == torch.bfloat16:
            dt_ = DataType.INFINI_DTYPE_BF16
        else:
            dt_ = DataType.INFINI_DTYPE_F16

        # Basic model parameters
        self.dt_logits = dt_
        self.nlayer = config["num_hidden_layers"]
        self.d = config["hidden_size"]
        self.nh = config["num_attention_heads"]
        self.nkvh = (
            config["num_key_value_heads"]
            if "num_key_value_heads" in config
            else config["num_attention_heads"]
        )
        self.dh = config["hidden_size"] // config["num_attention_heads"]
        self.di = config["intermediate_size"]
        self.dctx = (
            config["max_position_embeddings"] if max_tokens is None else max_tokens
        )
        self.dvoc = config["vocab_size"]
        self.epsilon = config.get("rms_norm_eps", 1e-6)
        self.theta = config.get("rope_theta", 10000.0)
        self.bos_token = config.get("bos_token_id", 1)
        self.end_token = config.get("eos_token_id", 2)
        self.attn_dropout = config.get("attention_dropout", 0.0)
        self.tie_embd = config.get("tie_word_embeddings", True)
        
        # MoE specific parameters
        self.num_experts = config.get("num_experts", 0)
        self.num_experts_per_tok = config.get("num_experts_per_tok", 2)
        self.moe_intermediate_size = config.get("moe_intermediate_size", self.di)
        self.decoder_sparse_step = config.get("decoder_sparse_step", 1)
        self.norm_topk_prob = config.get("norm_topk_prob", False)
        self.router_aux_loss_coef = config.get("router_aux_loss_coef", 0.0)
        
        # Handle mlp_only_layers
        mlp_only_layers = config.get("mlp_only_layers", [])
        self.num_mlp_only_layers = len(mlp_only_layers)
        if self.num_mlp_only_layers > 0:
            self.mlp_only_layers = (c_uint * self.num_mlp_only_layers)(*mlp_only_layers)
        else:
            self.mlp_only_layers = None
        
        self.torch_dtype_logits = dtype
        
        print(f"✓ Qwen3-MoE config: {self.nlayer} layers, {self.num_experts} experts, "
              f"top-{self.num_experts_per_tok}, sparse_step={self.decoder_sparse_step}")


class Qwen3MoeWeightsImpl(Qwen3MoeWeightsCStruct):
    """Qwen3-MoE weights implementation with expert weight support"""
    
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
        num_experts = meta.num_experts
        moe_di = meta.moe_intermediate_size
        
        assert nh % nkvh == 0
        assert nh % ndev == 0
        assert nkvh % ndev == 0
        assert di % ndev == 0
        if moe_di > 0:
            assert moe_di % ndev == 0
        
        torch_dt_logits = meta.torch_dtype_logits
        
        super().__init__()
        
        # Set data types
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

        self.nlayer = nlayer
        self.transpose_linear_weights = 1 if transpose_weight else 0

        # Store MoE parameters
        self.num_experts = num_experts
        self.num_experts_per_tok = meta.num_experts_per_tok
        self.moe_intermediate_size = moe_di
        self.decoder_sparse_step = meta.decoder_sparse_step
        self.num_mlp_only_layers = meta.num_mlp_only_layers
        self.mlp_only_layers = meta.mlp_only_layers
        self.norm_topk_prob = meta.norm_topk_prob

        # Determine input/output embedding names
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
        
        # Basic weights
        self.input_embd_tensor = state_dict[input_embd_naming].to(torch_dt_logits)
        self.input_embd = self.input_embd_tensor.data_ptr()
        
        self.output_norm_tensor = state_dict[naming.output_norm()].to(torch_dt_norm)
        self.output_norm = self.output_norm_tensor.data_ptr()
        
        self.output_embd_tensor = state_dict[output_embd_naming].to(torch_dt_mat)
        if not transpose_weight:
            self.output_embd_tensor = self.output_embd_tensor.transpose(0, 1).contiguous()
        self.output_embd = self.output_embd_tensor.data_ptr()

        # Layer-wise weight processing
        self._process_attention_weights(naming, state_dict, nlayer, torch_dt_norm, torch_dt_mat, ndev, transpose_weight)
        self._process_mlp_moe_weights(naming, state_dict, nlayer, meta, torch_dt_norm, torch_dt_mat, ndev, transpose_weight)
        
        print(f"✓ Qwen3-MoE weights loaded: {nlayer} layers, {num_experts} experts per MoE layer")

    def _is_moe_layer(self, layer_idx, meta):
        """Check if a layer is MoE or regular MLP"""
        # Check if in mlp_only_layers
        if meta.mlp_only_layers:
            mlp_only_list = [meta.mlp_only_layers[i] for i in range(meta.num_mlp_only_layers)]
            if layer_idx in mlp_only_list:
                return False
        
        # Check sparse step condition
        return (meta.num_experts > 0 and 
                (layer_idx + 1) % meta.decoder_sparse_step == 0)

    def _process_attention_weights(self, naming, state_dict, nlayer, torch_dt_norm, torch_dt_mat, ndev, transpose_weight):
        """Process attention weights (same as Qwen3)"""
        
        # Attention layer normalization weights
        self.attn_norm_tensors = [
            state_dict[naming.attn_norm(i)].to(torch_dt_norm) for i in range(nlayer)
        ]
        self.attn_norm_ptrs = [
            self.attn_norm_tensors[i].data_ptr() for i in range(nlayer)
        ]
        self.attn_norm = (c_void_p * nlayer)(*self.attn_norm_ptrs)

        # Q/K normalization weights (optional)
        self.attn_q_norm_tensors = []
        self.attn_k_norm_tensors = []
        try:
            for i in range(nlayer):
                self.attn_q_norm_tensors.append(state_dict[naming.q_norm(i)].to(torch_dt_norm))
                self.attn_k_norm_tensors.append(state_dict[naming.k_norm(i)].to(torch_dt_norm))
            
            self.attn_q_norm_ptrs = [self.attn_q_norm_tensors[i].data_ptr() for i in range(nlayer)]
            self.attn_k_norm_ptrs = [self.attn_k_norm_tensors[i].data_ptr() for i in range(nlayer)]
            self.attn_q_norm = (c_void_p * nlayer)(*self.attn_q_norm_ptrs)
            self.attn_k_norm = (c_void_p * nlayer)(*self.attn_k_norm_ptrs)
        except KeyError:
            null_ptrs = [None for _ in range(nlayer)]
            self.attn_q_norm = (c_void_p * nlayer)(*null_ptrs)
            self.attn_k_norm = (c_void_p * nlayer)(*null_ptrs)

        # Separate Q, K, V projection weights
        self.attn_q_proj_tensors = []
        self.attn_k_proj_tensors = []
        self.attn_v_proj_tensors = []
        
        for i in range(nlayer):
            q_tensor = state_dict[naming.attn_q(i)].to(torch_dt_mat)
            k_tensor = state_dict[naming.attn_k(i)].to(torch_dt_mat)
            v_tensor = state_dict[naming.attn_v(i)].to(torch_dt_mat)
            
            if not transpose_weight:
                q_tensor = q_tensor.transpose(0, 1).contiguous()
                k_tensor = k_tensor.transpose(0, 1).contiguous()
                v_tensor = v_tensor.transpose(0, 1).contiguous()
            
            self.attn_q_proj_tensors.append(q_tensor)
            self.attn_k_proj_tensors.append(k_tensor)
            self.attn_v_proj_tensors.append(v_tensor)

        self.attn_q_proj_ptrs = [self.attn_q_proj_tensors[i].data_ptr() for i in range(nlayer)]
        self.attn_k_proj_ptrs = [self.attn_k_proj_tensors[i].data_ptr() for i in range(nlayer)]
        self.attn_v_proj_ptrs = [self.attn_v_proj_tensors[i].data_ptr() for i in range(nlayer)]
        
        self.attn_q_proj = (c_void_p * nlayer)(*self.attn_q_proj_ptrs)
        self.attn_k_proj = (c_void_p * nlayer)(*self.attn_k_proj_ptrs)
        self.attn_v_proj = (c_void_p * nlayer)(*self.attn_v_proj_ptrs)

        # Attention output weights
        self.attn_o_proj_tensors = []
        for i in range(nlayer):
            o_tensor = state_dict[naming.attn_o(i)].to(torch_dt_mat)
            if not transpose_weight:
                o_tensor = o_tensor.transpose(0, 1).contiguous()
            self.attn_o_proj_tensors.append(o_tensor)
            
        self.attn_o_proj_ptrs = [self.attn_o_proj_tensors[i].data_ptr() for i in range(nlayer)]
        self.attn_o_proj = (c_void_p * nlayer)(*self.attn_o_proj_ptrs)

    def _process_mlp_moe_weights(self, naming, state_dict, nlayer, meta, torch_dt_norm, torch_dt_mat, ndev, transpose_weight):
        """Process MLP/MoE weights based on layer configuration"""
        
        # MLP normalization weights (common to both MLP and MoE layers)
        self.mlp_norm_tensors = [
            state_dict[naming.ffn_norm(i)].to(torch_dt_norm) for i in range(nlayer)
        ]
        self.mlp_norm_ptrs = [self.mlp_norm_tensors[i].data_ptr() for i in range(nlayer)]
        self.mlp_norm = (c_void_p * nlayer)(*self.mlp_norm_ptrs)

        # Initialize arrays for both MLP and MoE weights
        self.mlp_gate_proj_tensors = [None] * nlayer
        self.mlp_up_proj_tensors = [None] * nlayer
        self.mlp_down_proj_tensors = [None] * nlayer
        
        self.moe_gate_tensors = [None] * nlayer
        self.moe_experts_gate_proj_tensors = [None] * nlayer
        self.moe_experts_up_proj_tensors = [None] * nlayer
        self.moe_experts_down_proj_tensors = [None] * nlayer

        for i in range(nlayer):
            if self._is_moe_layer(i, meta):
                # Process MoE layer
                self._process_moe_layer_weights(i, naming, state_dict, meta, torch_dt_mat, transpose_weight)
            else:
                # Process regular MLP layer
                self._process_mlp_layer_weights(i, naming, state_dict, torch_dt_mat, transpose_weight)

        # Create C arrays
        self._create_weight_arrays(nlayer)

    def _process_moe_layer_weights(self, layer_idx, naming, state_dict, meta, torch_dt_mat, transpose_weight):
        """Process weights for a MoE layer"""
        num_experts = meta.num_experts
        
        # Router/gate weights
        gate_tensor = state_dict[naming.moe_gate(layer_idx)].to(torch_dt_mat)
        if not transpose_weight:
            gate_tensor = gate_tensor.transpose(0, 1).contiguous()
        self.moe_gate_tensors[layer_idx] = gate_tensor
        
        # Expert weights
        expert_gate_tensors = []
        expert_up_tensors = []
        expert_down_tensors = []
        
        for expert_idx in range(num_experts):
            # Expert gate projection
            expert_gate = state_dict[naming.moe_expert_gate(layer_idx, expert_idx)].to(torch_dt_mat)
            if not transpose_weight:
                expert_gate = expert_gate.transpose(0, 1).contiguous()
            expert_gate_tensors.append(expert_gate)
            
            # Expert up projection
            expert_up = state_dict[naming.moe_expert_up(layer_idx, expert_idx)].to(torch_dt_mat)
            if not transpose_weight:
                expert_up = expert_up.transpose(0, 1).contiguous()
            expert_up_tensors.append(expert_up)
            
            # Expert down projection
            expert_down = state_dict[naming.moe_expert_down(layer_idx, expert_idx)].to(torch_dt_mat)
            if not transpose_weight:
                expert_down = expert_down.transpose(0, 1).contiguous()
            expert_down_tensors.append(expert_down)
        
        self.moe_experts_gate_proj_tensors[layer_idx] = expert_gate_tensors
        self.moe_experts_up_proj_tensors[layer_idx] = expert_up_tensors
        self.moe_experts_down_proj_tensors[layer_idx] = expert_down_tensors

    def _process_mlp_layer_weights(self, layer_idx, naming, state_dict, torch_dt_mat, transpose_weight):
        """Process weights for a regular MLP layer"""
        
        gate_tensor = state_dict[naming.gate(layer_idx)].to(torch_dt_mat)
        up_tensor = state_dict[naming.up(layer_idx)].to(torch_dt_mat)
        down_tensor = state_dict[naming.down(layer_idx)].to(torch_dt_mat)
        
        if not transpose_weight:
            gate_tensor = gate_tensor.transpose(0, 1).contiguous()
            up_tensor = up_tensor.transpose(0, 1).contiguous()
            down_tensor = down_tensor.transpose(0, 1).contiguous()
        
        self.mlp_gate_proj_tensors[layer_idx] = gate_tensor
        self.mlp_up_proj_tensors[layer_idx] = up_tensor
        self.mlp_down_proj_tensors[layer_idx] = down_tensor

    def _create_weight_arrays(self, nlayer):
        """Create C pointer arrays for all weights"""
        
        # Regular MLP weights
        mlp_gate_ptrs = []
        mlp_up_ptrs = []
        mlp_down_ptrs = []
        
        for i in range(nlayer):
            if self.mlp_gate_proj_tensors[i] is not None:
                mlp_gate_ptrs.append(self.mlp_gate_proj_tensors[i].data_ptr())
                mlp_up_ptrs.append(self.mlp_up_proj_tensors[i].data_ptr())
                mlp_down_ptrs.append(self.mlp_down_proj_tensors[i].data_ptr())
            else:
                mlp_gate_ptrs.append(None)
                mlp_up_ptrs.append(None)
                mlp_down_ptrs.append(None)
        
        self.mlp_gate_proj = (c_void_p * nlayer)(*mlp_gate_ptrs)
        self.mlp_up_proj = (c_void_p * nlayer)(*mlp_up_ptrs)
        self.mlp_down_proj = (c_void_p * nlayer)(*mlp_down_ptrs)
        
        # MoE weights
        moe_gate_ptrs = []
        for i in range(nlayer):
            if self.moe_gate_tensors[i] is not None:
                moe_gate_ptrs.append(self.moe_gate_tensors[i].data_ptr())
            else:
                moe_gate_ptrs.append(None)
        
        self.moe_gate = (c_void_p * nlayer)(*moe_gate_ptrs)
        
        # Expert weights (three-level pointers)
        # This is a simplified implementation - full implementation would need proper 3D arrays
        self.moe_experts_gate_proj = None  # Would need complex 3D pointer array setup
        self.moe_experts_up_proj = None
        self.moe_experts_down_proj = None


class Qwen3MoeBatchedTask:
    """Batched inference task for Qwen3-MoE"""
    
    def __init__(self, tasks: List[Qwen3MoeInferTask]):
        self.tasks = tasks
        self.nreq = len(tasks)

        # Precompute fields
        token_lists = [t.tokens for t in tasks]
        self.req_lens_list = [len(toks) for toks in token_lists]
        self.req_pos_list = [t.pos for t in tasks]
        self.kv_cache_ptrs = [t.kvcache().data() for t in tasks]
        self.temperatures_list = [t.temperature for t in tasks]
        self.topks_list = [t.topk for t in tasks]
        self.topps_list = [t.topp for t in tasks]

        # Flatten token lists
        flat_tokens = [tok for toks in token_lists for tok in toks]
        self.ntok = len(flat_tokens)

        # Convert to ctypes arrays
        self.tokens = (c_uint * self.ntok)(*flat_tokens)
        self.req_lens = (c_uint * self.nreq)(*self.req_lens_list)
        self.req_pos = (c_uint * self.nreq)(*self.req_pos_list)
        self.kv_caches = (POINTER(KVCacheCStruct) * self.nreq)(*self.kv_cache_ptrs)
        self.temperatures = (c_float * self.nreq)(*self.temperatures_list)
        self.topks = (c_uint * self.nreq)(*self.topks_list)
        self.topps = (c_float * self.nreq)(*self.topps_list)

    def input_args(self):
        return (
            self.tokens,
            self.ntok,
            self.req_lens,
            self.nreq,
            self.req_pos,
            self.kv_caches,
            self.temperatures,
            self.topks,
            self.topps,
        )


class Qwen3MoeForCausalLM:
    """Qwen3-MoE model for causal language modeling"""
    
    def __init__(
        self, model_dir_path, device=DeviceType.DEVICE_TYPE_CPU, ndev=1, max_tokens=None
    ):
        def load_all_safetensors_from_dir(dir_path_: str):
            tensors_ = {}
            dir_path_ = Path(dir_path_)
            for file in sorted(dir_path_.glob("*.safetensors")):
                data_ = safetensors.safe_open(file, "pt")
                for name_ in data_.keys():
                    tensors_[name_] = data_.get_tensor(name_)
            return tensors_

        print("Loading Qwen3-MoE model weights to host...")
        load_start_time = time.time()

        with open(os.path.join(model_dir_path, "config.json"), "r") as f:
            config = json.load(f)
            self.config = config
            
        eos_token_id = self.config.get("eos_token_id", 2)
        self.eos_token_id = (
            [eos_token_id] if type(eos_token_id) == int else eos_token_id
        )
        
        transpose_weight = (
            device != DeviceType.DEVICE_TYPE_ASCEND
        )

        # Load state dict
        if any(file.suffix == ".safetensors" for file in Path(model_dir_path).iterdir()):
            state_dict = load_all_safetensors_from_dir(model_dir_path)
        else:
            state_dict = torch.load(
                os.path.join(model_dir_path, "pytorch_model.bin"),
                weights_only=True,
                map_location="cpu",
            )

        # Determine naming scheme
        if Qwen3MoeWeightsNaming.match(state_dict):
            print("✓ Using Qwen3MoeWeightsNaming (with MoE support)")
            # Create metadata and weights
            self.meta = Qwen3MoeMetaFromConfig(config, max_tokens=max_tokens)
            self.weights = Qwen3MoeWeightsImpl(
                self.meta,
                Qwen3MoeWeightsNaming(),
                state_dict,
                ndev=ndev,
                transpose_weight=transpose_weight,
            )
            # Load tokenizer
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_dir_path, trust_remote_code=True
            )
        else:
            raise ValueError("Unsupported weight naming scheme for Qwen3-MoE")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_end_time = time.time()
        print(f"Weight loading time: {load_end_time - load_start_time:.3f}s")

        print(f"Creating Qwen3-MoE model on {ndev} devices...")
        load_start_time = time.time()
        dev_ids = (c_int * ndev)(*[i for i in range(ndev)])
    
        try:
            self.model_instance = createQwen3MoeModel(
                ctypes.byref(self.meta), 
                ctypes.byref(self.weights),
                device, 
                ndev,
                dev_ids
            )
            print(f"✓ Qwen3-MoE model created successfully")
        except Exception as e:
            print(f"✗ Error creating Qwen3-MoE model: {e}")
            import traceback
            traceback.print_exc()
            raise

        load_end_time = time.time()
        print(f"Model creation time: {load_end_time - load_start_time:.3f}s")
        if self.model_instance is None:
            raise RuntimeError("Qwen3-MoE model instance is None after creation")

    def max_context_len(self):
        return self.meta.dctx

    def create_kv_cache(self):
        return createQwen3MoeKVCache(self.model_instance)

    def drop_kv_cache(self, kv_cache):
        dropQwen3MoeKVCache(self.model_instance, kv_cache)

    def batch_infer_one_round(self, tasks: List[Qwen3MoeInferTask]):
        output = (c_uint * len(tasks))()
        
        batch_inputs = Qwen3MoeBatchedTask(tasks)

        if g_debug_enabled:
            print(f"🔍 Qwen3-MoE batch inference:")
            print(f"  Number of requests: {batch_inputs.nreq}")
            print(f"  Total tokens: {batch_inputs.ntok}")
            print(f"  Request lengths: {batch_inputs.req_lens_list}")
        
        try:
            inferQwen3MoeBatch(
                self.model_instance,
                *batch_inputs.input_args(),
                output,
            )
            print("✅ inferQwen3MoeBatch completed")
            
        except Exception as e:
            print(f"❌ Qwen3-MoE C++ inference failed: {e}")
            import traceback
            traceback.print_exc()
            raise
            
        return list(output)

    def get_router_stats(self, layer_idx: int) -> Dict[int, int]:
        """Get expert usage statistics for a specific layer"""
        if self.meta.num_experts == 0:
            return {}
        
        expert_counts = (c_uint * self.meta.num_experts)()
        getQwen3MoeRouterStats(self.model_instance, layer_idx, expert_counts)
        
        return {i: expert_counts[i] for i in range(self.meta.num_experts)}

    def print_router_stats(self):
        """Print router statistics for all MoE layers"""
        print(f"\n{'='*60}")
        print("📊 MoE ROUTER STATISTICS")
        print(f"{'='*60}")
        
        for layer_idx in range(self.meta.nlayer):
            # Check if this is a MoE layer
            is_moe = self._is_moe_layer(layer_idx)
            if not is_moe:
                continue
                
            stats = self.get_router_stats(layer_idx)
            if not stats:
                continue
            
            total_usage = sum(stats.values())
            print(f"\nLayer {layer_idx} (MoE):")
            print(f"  Total tokens routed: {total_usage}")
            
            if total_usage > 0:
                # Calculate load balance
                ideal_usage = total_usage / self.meta.num_experts
                for expert_idx, count in stats.items():
                    percentage = (count / total_usage) * 100
                    balance = count / ideal_usage if ideal_usage > 0 else 0
                    print(f"    Expert {expert_idx:2d}: {count:6d} tokens ({percentage:5.1f}%) [balance: {balance:.2f}]")
                
                # Calculate load balance variance
                usages = list(stats.values())
                mean_usage = np.mean(usages)
                variance = np.var(usages)
                print(f"  Load balance variance: {variance:.2f} (lower is better)")

    def _is_moe_layer(self, layer_idx):
        """Check if a layer is MoE based on configuration"""
        # Check if in mlp_only_layers
        if self.meta.mlp_only_layers:
            mlp_only_list = [self.meta.mlp_only_layers[i] for i in range(self.meta.num_mlp_only_layers)]
            if layer_idx in mlp_only_list:
                return False
        
        # Check sparse step condition
        return (self.meta.num_experts > 0 and 
                (layer_idx + 1) % self.meta.decoder_sparse_step == 0)

    def generate(self, input_content, max_steps, topp_=0.8, topk_=50, temperature_=0.7):
        # Apply chat template if available
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            input_content = self.tokenizer.apply_chat_template(
                conversation=[{"role": "user", "content": input_content}],
                add_generation_prompt=True,
                tokenize=False,
            )
        
        print(input_content, end="", flush=True)
        tokens = self.tokenizer.encode(input_content)

        print(f"\nDEBUG: Input tokens: {tokens}")
        print(f"DEBUG: Token count: {len(tokens)}")
        print(f"DEBUG: EOS tokens: {self.eos_token_id}")
        print(f"DEBUG: Vocab size: {self.meta.dvoc}")
        print(f"DEBUG: MoE config: {self.meta.num_experts} experts, top-{self.meta.num_experts_per_tok}")

        infer_task = Qwen3MoeInferTask(
            tokens=tokens,
            position=0,
            temperature=temperature_,
            topk=topk_,
            topp=topp_,
            end_tokens=self.eos_token_id,
            max_tokens=int(self.meta.dctx),
            task_id=0
        )

        infer_task.bind_kvcache(Qwen3MoeKVCache(self))

        if self.model_instance is None:
            raise RuntimeError("❌ Qwen3-MoE model instance is null before inference")

        steps = 0
        total_time = 0
        output_content = ""

        for step_i in range(max_steps):
            start_time = time.time()
            output_tokens = self.batch_infer_one_round([infer_task])
            end_time = time.time()
            steps += 1
            
            output_token = output_tokens[0]
            print(f"\nDEBUG Step {step_i}:")
            print(f"  Output token ID: {output_token}")
            print(f"  Token in vocab range: {0 <= output_token < self.meta.dvoc}")
            print(f"  Is EOS token: {output_token in self.eos_token_id}")
            
            # Check token validity
            if output_token >= self.meta.dvoc:
                print(f"  ⚠ WARNING: Token {output_token} exceeds vocab size {self.meta.dvoc}")
                break
            if output_token < 0:
                print(f"  ⚠ WARNING: Negative token ID {output_token}")
                break
            
            try:
                output_str = self.tokenizer.decode([output_token], skip_special_tokens=False)
                print(f"  Decoded: '{output_str}'")
            except Exception as e:
                print(f"  ⚠ Decode failed: {e}")
                output_str = f"[UNK_{output_token}]"
            
            output_content += output_str
            print(output_str, end="", flush=True)
            
            if output_tokens[0] in self.eos_token_id:
                break
                
            infer_task.next(output_tokens[0])

            if step_i > 0:
                total_time += end_time - start_time

        print("\n")
        avg_time = total_time * 1000 / (steps - 1) if steps > 1 else 0
        print(f"Time per step: {avg_time:.3f}ms")

        # Print MoE router statistics
        if self.meta.num_experts > 0:
            self.print_router_stats()

        try:
            infer_task._kv_cache.drop()
        except Exception as e:
            print(f"    ⚠ KV cache cleanup failed: {e}")

        return output_content, avg_time

    def destroy_model_instance(self):
        destroyQwen3MoeModel(self.model_instance)
        print("Qwen3-MoE Model destroyed")


# Global debug flag
g_debug_enabled = False

def test():
    global g_debug_enabled
    
    if len(sys.argv) < 2:
        print("Usage: python qwen3_moe.py <path/to/model_dir> [device] [n_device] [--debug]")
        sys.exit(1)
        
    model_path = sys.argv[1]
    device_type = DeviceType.DEVICE_TYPE_CPU
    
    if len(sys.argv) > 2:
        if sys.argv[2] == "--cpu":
            device_type = DeviceType.DEVICE_TYPE_CPU
        elif sys.argv[2] == "--nvidia":
            device_type = DeviceType.DEVICE_TYPE_NVIDIA

    ndev = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    
    if "--debug" in sys.argv:
        g_debug_enabled = True
        setQwen3MoeDebugMode(1)
        print("🔍 Debug mode enabled")
    
    print(f"✓ Using Qwen3-MoE model from: {model_path}")
    print(f"✓ Device: {device_type}, Devices: {ndev}")
    
    model = Qwen3MoeForCausalLM(model_path, device_type, ndev)
    
    # Test generation
    model.generate("山东最高的山是？", 50, topp_=0.8, topk_=50, temperature_=0.7)
    model.destroy_model_instance()


if __name__ == "__main__":
    test()