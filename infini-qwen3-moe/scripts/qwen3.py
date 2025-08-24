#!/usr/bin/env python3
"""
Fixed Qwen3 Implementation - Using New qw C++ API
This version properly implements Qwen3 support using the dedicated qwen3 C++ API
with proper Q/K normalization and parameter mapping.

Key Improvements:
1. Uses dedicated Qwen3 API instead of fallback jiuge API
2. Handles Q/K normalization weights properly
3. Implements separate QKV projections
4. One-to-one parameter mapping following jiuge.py patterns
"""

from typing import List, Optional
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

# Set default device
torch.set_default_device("cpu")

# Import the proper Qwen3 API
try:
    from libinfinicore_infer import (
        Qwen3MetaCStruct,
        Qwen3WeightsCStruct,
        create_qwen3_model,
        destroy_qwen3_model,
        create_qwen3_kv_cache,
        drop_qwen3_kv_cache,
        infer_qwen3_batch,
        DataType,
        DeviceType,
        KVCacheCStruct,
    )
    QWEN3_API_AVAILABLE = True
    print("✓ Qwen3 C++ API available")
except ImportError as e:
    print(f"⚠ Qwen3 C++ API not available: {e}")
    print("  This version requires the qw implementation")
    sys.exit(1)

from infer_task import Qwen3InferTask, Qwen3KVCache


class Qwen3WeightsNaming:
    """Qwen3-specific weight naming with Q/K normalization and separate QKV support"""
    
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

    @staticmethod
    def match(state_dict):
        """Check if state_dict matches Qwen3 naming pattern"""
        has_basic = (
            "model.norm.weight" in state_dict
            and "model.layers.0.self_attn.q_proj.weight" in state_dict
        )
        # Qwen3 often has q_norm and k_norm weights
        has_qk_norm = (
            "model.layers.0.self_attn.q_norm.weight" in state_dict
            and "model.layers.0.self_attn.k_norm.weight" in state_dict
        )
        return has_basic and has_qk_norm

class Qwen3MetaFromConfig(Qwen3MetaCStruct):
    """Qwen3 metadata structure from model config"""
    
    def __init__(self, config, dtype=torch.float16, max_tokens=None):
        super().__init__()  # 先调用父类构造函数
        
        if dtype == torch.float16:
            dt_ = DataType.INFINI_DTYPE_F16
        elif dtype == torch.float32:
            dt_ = DataType.INFINI_DTYPE_F32
        elif dtype == torch.bfloat16:
            dt_ = DataType.INFINI_DTYPE_BF16
        else:
            dt_ = DataType.INFINI_DTYPE_F16

        # 设置字段值
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
        
        self.torch_dtype_logits = dtype

class Qwen3WeightsImpl(Qwen3WeightsCStruct):
    """Qwen3 weights implementation with Q/K normalization support"""
    
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
        
        assert nh % nkvh == 0
        assert nh % ndev == 0
        assert nkvh % ndev == 0
        assert di % ndev == 0
        
        torch_dt_logits = meta.torch_dtype_logits
        
        # 调用父类构造函数
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

        # 设置结构体字段
        self.nlayer = nlayer
        self.transpose_linear_weights = 1 if transpose_weight else 0

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

        # Attention layer normalization weights
        self.attn_norm_tensors = [
            state_dict[naming.attn_norm(i)].to(torch_dt_norm) for i in range(nlayer)
        ]
        self.attn_norm_ptrs = [
            self.attn_norm_tensors[i].data_ptr() for i in range(nlayer)
        ]
        self.attn_norm = (c_void_p * nlayer)(*self.attn_norm_ptrs)

        # Q/K normalization weights
        self.attn_q_norm_tensors = []
        self.attn_k_norm_tensors = []
        if hasattr(naming, 'q_norm'):
            try:
                for i in range(nlayer):
                    self.attn_q_norm_tensors.append(state_dict[naming.q_norm(i)].to(torch_dt_norm))
                    self.attn_k_norm_tensors.append(state_dict[naming.k_norm(i)].to(torch_dt_norm))
                
                self.attn_q_norm_ptrs = [self.attn_q_norm_tensors[i].data_ptr() for i in range(nlayer)]
                self.attn_k_norm_ptrs = [self.attn_k_norm_tensors[i].data_ptr() for i in range(nlayer)]
                self.attn_q_norm = (c_void_p * nlayer)(*self.attn_q_norm_ptrs)
                self.attn_k_norm = (c_void_p * nlayer)(*self.attn_k_norm_ptrs)
                
                print(f"✓ Loaded Q/K normalization weights for {nlayer} layers")
            except KeyError as e:
                print(f"⚠ Q/K norm weights not found: {e}")
                # 创建空指针数组
                null_ptrs = [None for _ in range(nlayer)]
                self.attn_q_norm = (c_void_p * nlayer)(*null_ptrs)
                self.attn_k_norm = (c_void_p * nlayer)(*null_ptrs)
        else:
            # 创建空指针数组
            null_ptrs = [None for _ in range(nlayer)]
            self.attn_q_norm = (c_void_p * nlayer)(*null_ptrs)
            self.attn_k_norm = (c_void_p * nlayer)(*null_ptrs)

        # 分离的 Q, K, V 投影权重
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

        # FFN weights
        self.mlp_norm_tensors = [
            state_dict[naming.ffn_norm(i)].to(torch_dt_norm) for i in range(nlayer)
        ]
        self.mlp_norm_ptrs = [self.mlp_norm_tensors[i].data_ptr() for i in range(nlayer)]
        self.mlp_norm = (c_void_p * nlayer)(*self.mlp_norm_ptrs)

        # 分离的 gate 和 up 投影
        self.mlp_gate_proj_tensors = []
        self.mlp_up_proj_tensors = []
        self.mlp_down_proj_tensors = []
        
        for i in range(nlayer):
            gate_tensor = state_dict[naming.gate(i)].to(torch_dt_mat)
            up_tensor = state_dict[naming.up(i)].to(torch_dt_mat)
            down_tensor = state_dict[naming.down(i)].to(torch_dt_mat)
            
            if not transpose_weight:
                gate_tensor = gate_tensor.transpose(0, 1).contiguous()
                up_tensor = up_tensor.transpose(0, 1).contiguous()
                down_tensor = down_tensor.transpose(0, 1).contiguous()
            
            self.mlp_gate_proj_tensors.append(gate_tensor)
            self.mlp_up_proj_tensors.append(up_tensor)
            self.mlp_down_proj_tensors.append(down_tensor)

        self.mlp_gate_proj_ptrs = [self.mlp_gate_proj_tensors[i].data_ptr() for i in range(nlayer)]
        self.mlp_up_proj_ptrs = [self.mlp_up_proj_tensors[i].data_ptr() for i in range(nlayer)]
        self.mlp_down_proj_ptrs = [self.mlp_down_proj_tensors[i].data_ptr() for i in range(nlayer)]
        
        self.mlp_gate_proj = (c_void_p * nlayer)(*self.mlp_gate_proj_ptrs)
        self.mlp_up_proj = (c_void_p * nlayer)(*self.mlp_up_proj_ptrs)
        self.mlp_down_proj = (c_void_p * nlayer)(*self.mlp_down_proj_ptrs)

        # 验证所有关键权重都已加载
        required_weights = [
            self.input_embd_tensor,
            self.output_embd_tensor,
            self.output_norm_tensor,
        ]
        
        for i, tensor in enumerate(required_weights):
            if tensor is None or tensor.data_ptr() == 0:
                raise RuntimeError(f"Critical weight {i} is None or has null data pointer")
        
        # 验证层权重
        for i in range(nlayer):
            critical_tensors = [
                self.attn_norm_tensors[i],
                self.attn_q_proj_tensors[i],
                self.attn_k_proj_tensors[i],
                self.attn_v_proj_tensors[i],
                self.attn_o_proj_tensors[i],
                self.mlp_norm_tensors[i],
                self.mlp_gate_proj_tensors[i],
                self.mlp_up_proj_tensors[i],
                self.mlp_down_proj_tensors[i],
            ]
            
            for j, tensor in enumerate(critical_tensors):
                if tensor is None or tensor.data_ptr() == 0:
                    raise RuntimeError(f"Layer {i} weight {j} is None or has null data pointer")
        
        print(f"✓ All {nlayer} layers' weights validated successfully")
        print("🔍 First 10 values of attn_q_0:")
        print(self.attn_q_proj_tensors[0].flatten()[:10])
        print("🔍 Pointer address:", hex(self.attn_q_proj_tensors[0].data_ptr()))

class Qwen3BatchedTask:
    """Batched inference task for Qwen3"""
    
    def __init__(self, tasks: List[Qwen3InferTask]):
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


class QwenForCausalLM:
    """Qwen3 model for causal language modeling - FIXED VERSION"""
    
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

        print("Loading Qwen3 model weights to host...")
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
        if Qwen3WeightsNaming.match(state_dict):
            print("✓ Using Qwen3WeightsNaming (with Q/K normalization)")
                    # Create metadata and weights
            self.meta = Qwen3MetaFromConfig(config, max_tokens=max_tokens)
            self.weights = Qwen3WeightsImpl(
                self.meta,
                Qwen3WeightsNaming(),
                state_dict,
                ndev=ndev,
                transpose_weight=transpose_weight,
            )
            # Load tokenizer
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_dir_path, trust_remote_code=True
            )
        elif LlamaWeightsNaming.match(state_dict):
            print("⚠ Using LlamaWeightsNaming (fallback, no Q/K normalization)")
        else:
            raise ValueError("Unsupported weight naming scheme")




        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_end_time = time.time()
        print(f"Weight loading time: {load_end_time - load_start_time:.3f}s")

        print(f"Creating Qwen3 model on {ndev} devices...")
        load_start_time = time.time()
        dev_ids = (c_int * ndev)(*[i for i in range(ndev)])
    
        try:
            self.model_instance = create_qwen3_model(
                ctypes.byref(self.meta), 
                ctypes.byref(self.weights),
                device, 
                ndev,
                dev_ids
            )
            print(f"✓ Model created successfully")
        except Exception as e:
            print(f"✗ Error creating model: {e}")
            import traceback
            traceback.print_exc()
            raise

        load_end_time = time.time()
        print(f"Model creation time: {load_end_time - load_start_time:.3f}s")
        if self.model_instance is None:
            raise RuntimeError("Model instance is None after creation")

    def max_context_len(self):
        return self.meta.dctx

    def create_kv_cache(self):
        # FIXED: Use proper Qwen3 KV cache API
        return create_qwen3_kv_cache(self.model_instance)

    def drop_kv_cache(self, kv_cache):
        # FIXED: Use proper Qwen3 KV cache API
        drop_qwen3_kv_cache(self.model_instance, kv_cache)

    def batch_infer_one_round(self, tasks: List[Qwen3InferTask]):
        output = (c_uint * len(tasks))()
        
        # 使用Qwen3BatchedTask类来处理批处理
        batch_inputs = Qwen3BatchedTask(tasks)


        # 详细验证输入参数
        # print(f"DEBUG Batch inference:")
        # print(f"  Number of requests: {batch_inputs.nreq}")
        # print(f"  Total tokens: {batch_inputs.ntok}")
        # print(f"  Request lengths: {batch_inputs.req_lens_list}")
        # print(f"  Request positions: {batch_inputs.req_pos_list}")
        # print(f"  Temperatures: {batch_inputs.temperatures_list}")
        # print(f"  Top-k values: {batch_inputs.topks_list}")
        # print(f"  Top-p values: {batch_inputs.topps_list}")
        
        # 检查C++函数调用参数的内存地址 - FIXED
        print(f"🔍 C++ function parameters:")
        try:
            model_ptr_addr = ctypes.addressof(self.model_instance.contents) if self.model_instance else 0
            print(f"  model_instance: {hex(model_ptr_addr)}")
        except:
            print(f"  model_instance: exists={self.model_instance is not None}")
            
        print(f"  tokens array ptr: {hex(ctypes.addressof(batch_inputs.tokens))}")
        print(f"  kv_caches array ptr: {hex(ctypes.addressof(batch_inputs.kv_caches))}")
        print(f"  output array ptr: {hex(ctypes.addressof(output))}")

            # 检查输入token的合理性
        if batch_inputs.ntok > 0:
            first_few_tokens = list(batch_inputs.tokens)[:min(5, batch_inputs.ntok)]
            print(f"  First few input tokens: {first_few_tokens}")
        
        
        # 验证输入参数
        if batch_inputs.ntok == 0:
            raise ValueError("没有tokens需要处理")
        if batch_inputs.nreq == 0:
            raise ValueError("没有请求需要处理")
        
        try:
            # 使用batch_inputs中的数组
            print("🚀 Calling infer_qwen3_batch...")
            infer_qwen3_batch(
                self.model_instance,
                *batch_inputs.input_args(),
                output,
            )
            print("✅ infer_qwen3_batch completed")
            
            # 验证输出token
            for i, token in enumerate(list(output)):
                print(f"  Output token[{i}]: {token}")
                if token >= self.meta.dvoc:
                    print(f"    ⚠ Invalid: exceeds vocab_size {self.meta.dvoc}")
                if token < 0:
                    print(f"    ⚠ Invalid: negative token")
                    
        except Exception as e:
            print(f"❌ C++ inference failed: {e}")
            import traceback
            traceback.print_exc()
            raise
            
        return list(output)
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

        # 添加详细调试信息
        print(f"\nDEBUG: Input tokens: {tokens}")
        print(f"DEBUG: Token count: {len(tokens)}")
        print(f"DEBUG: EOS tokens: {self.eos_token_id}")
        print(f"DEBUG: Vocab size: {self.meta.dvoc}")

            # 验证输入token的合理性
        print("🔍 Validating input tokens:")
        for i, token in enumerate(tokens[:5]):  # 只检查前5个
            if token >= self.meta.dvoc or token < 0:
                raise ValueError(f"❌ Invalid input token[{i}]: {token} (vocab_size: {self.meta.dvoc})")
            
            # 检查token对应的embedding
            embd_vec = self.weights.input_embd_tensor[token]
            embd_norm = embd_vec.norm().item()
            print(f"    Token[{i}]={token}, embedding_norm={embd_norm:.6f}")
            
            if embd_norm < 1e-8:
                print(f"    ⚠ Warning: Token {token} has very small embedding norm")
            if torch.isnan(embd_vec).any() or torch.isinf(embd_vec).any():
                raise RuntimeError(f"❌ Token {token} embedding contains NaN/Inf")
            
        infer_task = Qwen3InferTask(
            tokens=tokens,
            position=0,
            temperature=temperature_,
            topk=topk_,
            topp=topp_,
            end_tokens=self.eos_token_id,
            max_tokens=int(self.meta.dctx),
            task_id=0
        )

        infer_task.bind_kvcache(Qwen3KVCache(self))
        # 验证模型实例状态 - FIXED
        print(f"🔍 Model instance validation:")
        try:
            model_ptr_addr = ctypes.addressof(self.model_instance.contents) if self.model_instance else 0
            print(f"    Model instance ptr: {hex(model_ptr_addr)}")
        except:
            # 降级处理
            print(f"    Model instance: {self.model_instance is not None}")
            
        if self.model_instance is None:
            raise RuntimeError("❌ Model instance is null before inference")


        steps = 0
        total_time = 0
        output_content = ""

        for step_i in range(max_steps):
            start_time = time.time()
            output_tokens = self.batch_infer_one_round([infer_task])
            end_time = time.time()
            steps += 1
                    # 详细的token分析
            output_token = output_tokens[0]
            print(f"\nDEBUG Step {step_i}:")
            print(f"  Output token ID: {output_token}")
            print(f"  Token in vocab range: {0 <= output_token < self.meta.dvoc}")
            print(f"  Is EOS token: {output_token in self.eos_token_id}")
            # 检查token合理性
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
                output_str = self.tokenizer._tokenizer.id_to_token(output_token)
                if output_str is None:
                    output_str = f"[UNK_{output_token}]"
                else:
                    output_str = output_str.replace("▁", " ").replace("<0x0A>", "\n")
            
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

        try:
            infer_task._kv_cache.drop()
        except AttributeError:
            # 如果drop方法有问题，跳过清理
            print("    ⚠ KV cache cleanup skipped (method issue)")
        except Exception as e:
            print(f"    ⚠ KV cache cleanup failed: {e}")

        return output_content, avg_time

    def destroy_model_instance(self):
        # FIXED: Use proper Qwen3 model destruction API
        destroy_qwen3_model(self.model_instance)
        print("Qwen3 Model destroyed")

    def diagnose_cpp_computation(self):
        """诊断C++推理引擎的计算正确性"""
        
        print(f"\n{'='*60}")
        print("🔬 C++ COMPUTATION DIAGNOSIS")
        print(f"{'='*60}")
        
        # 1. 测试固定输入的一致性
        print("\n1️⃣ Testing computation consistency with fixed inputs:")
        
        # 使用非常简单的输入
        simple_tokens = [1, 2, 3]  # BOS, simple tokens
        
        results = []
        for i in range(3):  # 运行3次相同的推理
            print(f"\n  Run {i+1}/3:")
            
            task = Qwen3InferTask(
                tokens=simple_tokens,
                position=0,
                temperature=0.0,  # 完全确定性
                topk=1,           # 只取概率最高的token
                topp=1.0,
                end_tokens=self.eos_token_id,
                max_tokens=int(self.meta.dctx),
                task_id=0
            )
            task.bind_kvcache(Qwen3KVCache(self))
            
            # 执行推理
            output_tokens = self.batch_infer_one_round([task])
            output_token = output_tokens[0]
            
            print(f"    Input tokens: {simple_tokens}")
            print(f"    Output token: {output_token}")
            
            results.append(output_token)
                    # FIXED: 使用try-except来处理KV缓存清理
            try:
                task._kv_cache.drop()
            except AttributeError:
                # 如果drop方法有问题，跳过清理
                print("    ⚠ KV cache cleanup skipped (method issue)")
            except Exception as e:
                print(f"    ⚠ KV cache cleanup failed: {e}")
        
        # 检查一致性
        if len(set(results)) == 1:
            print(f"  ✅ PASS: All runs produced same result: {results[0]}")
        else:
            print(f"  ❌ FAIL: Inconsistent results: {results}")
            print("    This indicates non-deterministic computation or memory corruption")
        
        # 2. 测试不同temperature的影响
        print("\n2️⃣ Testing temperature parameter effect:")
        
        temps = [0.0, 0.5, 1.0]
        temp_results = {}
        
        for temp in temps:
            task = Qwen3InferTask(
                tokens=simple_tokens,
                position=0,
                temperature=temp,
                topk=50,
                topp=0.8,
                end_tokens=self.eos_token_id,
                max_tokens=int(self.meta.dctx),
                task_id=0
            )
            task.bind_kvcache(Qwen3KVCache(self))
            
            # 运行多次获取分布
            temp_outputs = []
            for _ in range(5):
                output_tokens = self.batch_infer_one_round([task])
                temp_outputs.append(output_tokens[0])
                task.next(output_tokens[0])  # 更新状态以便下次推理
            
            temp_results[temp] = temp_outputs
            try:
                task._kv_cache.drop()
            except AttributeError:
                # 如果drop方法有问题，跳过清理
                print("    ⚠ KV cache cleanup skipped (method issue)")
            except Exception as e:
                print(f"    ⚠ KV cache cleanup failed: {e}")

            unique_outputs = len(set(temp_outputs))
            print(f"    temp={temp}: outputs={temp_outputs}, unique={unique_outputs}")
        
        # 验证temperature=0.0应该完全确定
        if len(set(temp_results[0.0])) == 1:
            print("  ✅ PASS: Temperature=0.0 produces deterministic output")
        else:
            print("  ❌ FAIL: Temperature=0.0 should be deterministic")
        
        # 3. 测试输入长度对输出的影响
        print("\n3️⃣ Testing input length effect:")
        
        test_inputs = [
            [1],           # 1 token
            [1, 2],        # 2 tokens  
            [1, 2, 3],     # 3 tokens
            [1, 2, 3, 4],  # 4 tokens
        ]
        
        length_results = {}
        for tokens in test_inputs:
            task = Qwen3InferTask(
                tokens=tokens,
                position=0,
                temperature=0.0,
                topk=1,
                topp=1.0,
                end_tokens=self.eos_token_id,
                max_tokens=int(self.meta.dctx),
                task_id=0
            )
            task.bind_kvcache(Qwen3KVCache(self))
            
            output_tokens = self.batch_infer_one_round([task])
            length_results[len(tokens)] = output_tokens[0]
            
            print(f"    Input length {len(tokens)}: {tokens} -> {output_tokens[0]}")
            try:
                task._kv_cache.drop()
            except AttributeError:
                # 如果drop方法有问题，跳过清理
                print("    ⚠ KV cache cleanup skipped (method issue)")
            except Exception as e:
                print(f"    ⚠ KV cache cleanup failed: {e}")

        # 4. 测试KV缓存状态的影响
        print("\n4️⃣ Testing KV cache state impact:")
        
        # 第一次推理
        task1 = Qwen3InferTask(
            tokens=[1, 2, 3],
            position=0,
            temperature=0.0,
            topk=1,
            topp=1.0,
            end_tokens=self.eos_token_id,
            max_tokens=int(self.meta.dctx),
            task_id=0
        )
        task1.bind_kvcache(Qwen3KVCache(self))
        
        output1 = self.batch_infer_one_round([task1])[0]
        print(f"    Fresh KV cache: [1,2,3] -> {output1}")
        
        # 继续用相同的KV缓存推理下一个token
        task1.next(output1)
        output2 = self.batch_infer_one_round([task1])[0]
        print(f"    Continued KV cache: append {output1} -> {output2}")
        
        # 重新开始，但用不同的方式
        task2 = Qwen3InferTask(
            tokens=[1, 2, 3, output1],
            position=0,
            temperature=0.0,
            topk=1,
            topp=1.0,
            end_tokens=self.eos_token_id,
            max_tokens=int(self.meta.dctx),
            task_id=0
        )
        task2.bind_kvcache(Qwen3KVCache(self))
        
        output3 = self.batch_infer_one_round([task2])[0]
        print(f"    Fresh KV cache: [1,2,3,{output1}] -> {output3}")
        
        if output2 == output3:
            print("  ✅ PASS: KV cache state consistency maintained")
        else:
            print("  ❌ FAIL: KV cache state inconsistent")
            print(f"         Continued: {output2}, Fresh: {output3}")
        
        task1._kv_cache.drop()
        task2._kv_cache.drop()
        
        # 5. 测试边界情况
        print("\n5️⃣ Testing edge cases:")
        
        # 测试vocab边界附近的token
        edge_tokens = [0, 1, self.meta.dvoc-2, self.meta.dvoc-1]  # 避免无效token
        
        for token in edge_tokens:
            if 0 <= token < self.meta.dvoc:
                task = Qwen3InferTask(
                    tokens=[token],
                    position=0,
                    temperature=0.0,
                    topk=1,
                    topp=1.0,
                    end_tokens=self.eos_token_id,
                    max_tokens=int(self.meta.dctx),
                    task_id=0
                )
                task.bind_kvcache(Qwen3KVCache(self))
                
                try:
                    output = self.batch_infer_one_round([task])[0]
                    print(f"    Edge token {token} -> {output}")
                    
                    if 0 <= output < self.meta.dvoc:
                        print(f"      ✅ Output {output} in valid range")
                    else:
                        print(f"      ❌ Output {output} out of range [0, {self.meta.dvoc})")
                        
                except Exception as e:
                    print(f"      ❌ Error with token {token}: {e}")
                finally:
                    try:
                        task._kv_cache.drop()
                    except AttributeError:
                        # 如果drop方法有问题，跳过清理
                        print("    ⚠ KV cache cleanup skipped (method issue)")
                    except Exception as e:
                        print(f"    ⚠ KV cache cleanup failed: {e}")

        print(f"\n{'='*60}")
        print("🔬 DIAGNOSIS COMPLETE")
        print(f"{'='*60}")
    
    def generate_simple(self, input_content, max_steps, topp_=0.8, topk_=50, temperature_=0.7):
        """不使用chat template的简单生成"""
        print(f"\nSimple generation: '{input_content}'", end="", flush=True)
        tokens = self.tokenizer.encode(input_content)
    
        print(f"\nInput tokens: {tokens}")
        
        infer_task = Qwen3InferTask(
            tokens=tokens,
            position=0,
            temperature=temperature_,
            topk=topk_,
            topp=topp_,
            end_tokens=self.eos_token_id,
            max_tokens=int(self.meta.dctx),
            task_id=0
        )
    
        infer_task.bind_kvcache(Qwen3KVCache(self))
    
        output_content = ""
        for step_i in range(max_steps):
            output_tokens = self.batch_infer_one_round([infer_task])
            output_token = output_tokens[0]
            
            print(f" -> {output_token}", end="")
            
            if output_token >= self.meta.dvoc or output_token < 0:
                print(f" (INVALID)")
                break
            
            try:
                output_str = self.tokenizer.decode([output_token], skip_special_tokens=False)
            except Exception:
                output_str = f"[UNK_{output_token}]"
            
            output_content += output_str
            print(f"('{output_str}')", end="", flush=True)
            
            if output_token in self.eos_token_id:
                break
                
            infer_task.next(output_token)
    
        print(f"\nFinal output: '{output_content}'")
        try:
            infer_task._kv_cache.drop()
        except AttributeError:
            # 如果drop方法有问题，跳过清理
            print("    ⚠ KV cache cleanup skipped (method issue)")
        except Exception as e:
            print(f"    ⚠ KV cache cleanup failed: {e}")
        return output_content
def test():
    if len(sys.argv) < 2:
        print("Usage: python qwen3_fixed.py <path/to/model_dir> [device] [n_device]")
        sys.exit(1)
        
    model_path = sys.argv[1]
    device_type = DeviceType.DEVICE_TYPE_CPU
    
    if len(sys.argv) > 2:
        if sys.argv[2] == "--cpu":
            device_type = DeviceType.DEVICE_TYPE_CPU
        elif sys.argv[2] == "--nvidia":
            device_type = DeviceType.DEVICE_TYPE_NVIDIA

    ndev = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    
    print(f"✓ Using Qwen3 model from: {model_path}")
    print(f"✓ Device: {device_type}, Devices: {ndev}")
    
    model = QwenForCausalLM(model_path, device_type, ndev)
    
    # 诊断C++计算问题
    model.diagnose_cpp_computation()
    
    # 然后测试生成
    model.generate("山东最高的山是？", 5, topp_=0.8, topk_=50, temperature_=0.7)
    model.destroy_model_instance()


if __name__ == "__main__":
    test()