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
            dt_value = DataType.INFINI_DTYPE_F16
        elif dtype == torch.float32:
            dt_value = DataType.INFINI_DTYPE_F32
        elif dtype == torch.bfloat16:
            dt_value = DataType.INFINI_DTYPE_BF16
        else:
            dt_value = DataType.INFINI_DTYPE_F16

        # 设置字段值
        self.dt_logits = dt_value
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

class Qwen3WeightsImpl:
    """重新设计的Qwen3权重实现类，使用组合模式而非继承"""
    
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
        # 提取关键参数
        nlayer = meta.nlayer
        nh = meta.nh
        nkvh = meta.nkvh
        dh = meta.dh
        d = meta.d
        di = meta.di
        
        # 验证设备分布约束
        assert nh % nkvh == 0, f"注意力头数 {nh} 必须是KV头数 {nkvh} 的倍数"
        assert nh % ndev == 0, f"注意力头数 {nh} 必须可被设备数 {ndev} 整除"
        assert nkvh % ndev == 0, f"KV头数 {nkvh} 必须可被设备数 {ndev} 整除"
        assert di % ndev == 0, f"中间维度 {di} 必须可被设备数 {ndev} 整除"
        
        torch_dt_logits = meta.torch_dtype_logits
        
        # 创建C结构体，而非继承它
        self.c_struct = Qwen3WeightsCStruct()
        
        # 保存所有张量引用，防止被垃圾回收
        self._tensor_refs = []
        
        # 设置基本字段
        self.c_struct.nlayer = nlayer
        self.c_struct.transpose_linear_weights = 1 if transpose_weight else 0
        
       # 设置数据类型 - 修复：直接使用枚举值，不要包装
        if torch_dt_mat == torch.float16:
            self.c_struct.dt_mat = DataType.INFINI_DTYPE_F16
        elif torch_dt_mat == torch.float32:
            self.c_struct.dt_mat = DataType.INFINI_DTYPE_F32
        elif torch_dt_mat == torch.bfloat16:
            self.c_struct.dt_mat = DataType.INFINI_DTYPE_BF16
        else:
            raise ValueError("不支持的投影权重数据类型")
            
        if torch_dt_norm == torch.float16:
            self.c_struct.dt_norm = DataType.INFINI_DTYPE_F16
        elif torch_dt_norm == torch.float32:
            self.c_struct.dt_norm = DataType.INFINI_DTYPE_F32
        elif torch_dt_norm == torch.bfloat16:
            self.c_struct.dt_norm = DataType.INFINI_DTYPE_BF16
        else:
            raise ValueError("不支持的归一化权重数据类型")

        # 确定输入/输出嵌入名称
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
        
        # ---- 基础权重加载 ----
        # 输入嵌入
        input_embd_tensor = state_dict[input_embd_naming].to(torch_dt_logits)
        self._tensor_refs.append(input_embd_tensor)
        self.c_struct.input_embd = input_embd_tensor.data_ptr()
        
        # 输出归一化
        output_norm_tensor = state_dict[naming.output_norm()].to(torch_dt_norm)
        self._tensor_refs.append(output_norm_tensor)
        self.c_struct.output_norm = output_norm_tensor.data_ptr()
        
        # 输出嵌入
        output_embd_tensor = state_dict[output_embd_naming].to(torch_dt_mat)
        if not transpose_weight:
            output_embd_tensor = output_embd_tensor.transpose(0, 1).contiguous()
        self._tensor_refs.append(output_embd_tensor)
        self.c_struct.output_embd = output_embd_tensor.data_ptr()

        # ---- 注意力层归一化权重 ----
        attn_norm_tensors = [
            state_dict[naming.attn_norm(i)].to(torch_dt_norm) for i in range(nlayer)
        ]
        self._tensor_refs.extend(attn_norm_tensors)
        attn_norm_ptrs = [tensor.data_ptr() for tensor in attn_norm_tensors]
        self.c_struct.attn_norm = (c_void_p * nlayer)(*attn_norm_ptrs)

        # ---- Q/K归一化权重（Qwen3特有） ----
        attn_q_norm_tensors = []
        attn_k_norm_tensors = []
        
        if hasattr(naming, 'q_norm') and hasattr(naming, 'k_norm'):
            try:
                for i in range(nlayer):
                    q_norm_tensor = state_dict[naming.q_norm(i)].to(torch_dt_norm)
                    k_norm_tensor = state_dict[naming.k_norm(i)].to(torch_dt_norm)
                    attn_q_norm_tensors.append(q_norm_tensor)
                    attn_k_norm_tensors.append(k_norm_tensor)
                
                self._tensor_refs.extend(attn_q_norm_tensors)
                self._tensor_refs.extend(attn_k_norm_tensors)
                
                attn_q_norm_ptrs = [tensor.data_ptr() for tensor in attn_q_norm_tensors]
                attn_k_norm_ptrs = [tensor.data_ptr() for tensor in attn_k_norm_tensors]
                
                self.c_struct.attn_q_norm = (c_void_p * nlayer)(*attn_q_norm_ptrs)
                self.c_struct.attn_k_norm = (c_void_p * nlayer)(*attn_k_norm_ptrs)
                
                print(f"✓ 已加载{nlayer}层的Q/K归一化权重")
            except KeyError as e:
                print(f"⚠ 未找到Q/K归一化权重: {e}")
                # 创建空指针数组
                null_ptrs = [None for _ in range(nlayer)]
                self.c_struct.attn_q_norm = (c_void_p * nlayer)(*null_ptrs)
                self.c_struct.attn_k_norm = (c_void_p * nlayer)(*null_ptrs)
        else:
            # 创建空指针数组
            null_ptrs = [None for _ in range(nlayer)]
            self.c_struct.attn_q_norm = (c_void_p * nlayer)(*null_ptrs)
            self.c_struct.attn_k_norm = (c_void_p * nlayer)(*null_ptrs)

        # ---- QKV投影权重（分开存储） ----
        attn_q_proj_tensors = []
        attn_k_proj_tensors = []
        attn_v_proj_tensors = []
        
        for i in range(nlayer):
            q_tensor = state_dict[naming.attn_q(i)].to(torch_dt_mat)
            k_tensor = state_dict[naming.attn_k(i)].to(torch_dt_mat)
            v_tensor = state_dict[naming.attn_v(i)].to(torch_dt_mat)
            
            if not transpose_weight:
                q_tensor = q_tensor.transpose(0, 1).contiguous()
                k_tensor = k_tensor.transpose(0, 1).contiguous()
                v_tensor = v_tensor.transpose(0, 1).contiguous()
            
            attn_q_proj_tensors.append(q_tensor)
            attn_k_proj_tensors.append(k_tensor)
            attn_v_proj_tensors.append(v_tensor)

        self._tensor_refs.extend(attn_q_proj_tensors)
        self._tensor_refs.extend(attn_k_proj_tensors)
        self._tensor_refs.extend(attn_v_proj_tensors)
        
        attn_q_proj_ptrs = [tensor.data_ptr() for tensor in attn_q_proj_tensors]
        attn_k_proj_ptrs = [tensor.data_ptr() for tensor in attn_k_proj_tensors]
        attn_v_proj_ptrs = [tensor.data_ptr() for tensor in attn_v_proj_tensors]
        
        self.c_struct.attn_q_proj = (c_void_p * nlayer)(*attn_q_proj_ptrs)
        self.c_struct.attn_k_proj = (c_void_p * nlayer)(*attn_k_proj_ptrs)
        self.c_struct.attn_v_proj = (c_void_p * nlayer)(*attn_v_proj_ptrs)

        # ---- 注意力输出权重 ----
        attn_o_proj_tensors = []
        for i in range(nlayer):
            o_tensor = state_dict[naming.attn_o(i)].to(torch_dt_mat)
            if not transpose_weight:
                o_tensor = o_tensor.transpose(0, 1).contiguous()
            attn_o_proj_tensors.append(o_tensor)
        
        self._tensor_refs.extend(attn_o_proj_tensors)
        attn_o_proj_ptrs = [tensor.data_ptr() for tensor in attn_o_proj_tensors]
        self.c_struct.attn_o_proj = (c_void_p * nlayer)(*attn_o_proj_ptrs)

        # ---- FFN归一化权重 ----
        mlp_norm_tensors = [
            state_dict[naming.ffn_norm(i)].to(torch_dt_norm) for i in range(nlayer)
        ]
        self._tensor_refs.extend(mlp_norm_tensors)
        mlp_norm_ptrs = [tensor.data_ptr() for tensor in mlp_norm_tensors]
        self.c_struct.mlp_norm = (c_void_p * nlayer)(*mlp_norm_ptrs)

        # ---- FFN投影权重（分开存储） ----
        mlp_gate_proj_tensors = []
        mlp_up_proj_tensors = []
        mlp_down_proj_tensors = []
        
        for i in range(nlayer):
            gate_tensor = state_dict[naming.gate(i)].to(torch_dt_mat)
            up_tensor = state_dict[naming.up(i)].to(torch_dt_mat)
            down_tensor = state_dict[naming.down(i)].to(torch_dt_mat)
            
            if not transpose_weight:
                gate_tensor = gate_tensor.transpose(0, 1).contiguous()
                up_tensor = up_tensor.transpose(0, 1).contiguous()
                down_tensor = down_tensor.transpose(0, 1).contiguous()
            
            mlp_gate_proj_tensors.append(gate_tensor)
            mlp_up_proj_tensors.append(up_tensor)
            mlp_down_proj_tensors.append(down_tensor)

        self._tensor_refs.extend(mlp_gate_proj_tensors)
        self._tensor_refs.extend(mlp_up_proj_tensors)
        self._tensor_refs.extend(mlp_down_proj_tensors)
        
        mlp_gate_proj_ptrs = [tensor.data_ptr() for tensor in mlp_gate_proj_tensors]
        mlp_up_proj_ptrs = [tensor.data_ptr() for tensor in mlp_up_proj_tensors]
        mlp_down_proj_ptrs = [tensor.data_ptr() for tensor in mlp_down_proj_tensors]
        
        self.c_struct.mlp_gate_proj = (c_void_p * nlayer)(*mlp_gate_proj_ptrs)
        self.c_struct.mlp_up_proj = (c_void_p * nlayer)(*mlp_up_proj_ptrs)
        self.c_struct.mlp_down_proj = (c_void_p * nlayer)(*mlp_down_proj_ptrs)

        # ---- 验证和修复流程 ----
        # 确保张量连续
        self.ensure_tensors_contiguous()
        
        # 验证张量数据
        self.validate_tensor_data()
        
        # ---- 验证关键权重 ----
        self.validate_weights()
        
    def ensure_tensors_contiguous(self):
        """确保所有张量是连续内存布局"""
        print("\n检查张量内存连续性...")
        non_contiguous_count = 0
        
        for i, tensor in enumerate(self._tensor_refs):
            if not tensor.is_contiguous():
                print(f"  警告: 张量 {i} 不连续")
                self._tensor_refs[i] = tensor.contiguous()  # 替换为连续版本
                non_contiguous_count += 1
        
        if non_contiguous_count > 0:
            print(f"  已修复 {non_contiguous_count} 个不连续张量")
        else:
            print("  所有张量已是连续内存布局")
    
    def validate_tensor_data(self):
        """验证张量数据有效性"""
        print("\n==== 张量数据验证 ====")
        
        # 检查常见问题，如NaN和Inf
        has_issues = False
        for i, tensor in enumerate(self._tensor_refs[:10]):  # 检查前10个张量
            # 检查NaN
            if torch.isnan(tensor).any():
                print(f"  ❌ 张量 {i} 包含NaN值")
                has_issues = True
            
            # 检查Inf
            if torch.isinf(tensor).any():
                print(f"  ❌ 张量 {i} 包含Inf值")
                has_issues = True
                
            # 检查是否全零
            if (tensor == 0).all():
                print(f"  ⚠️ 张量 {i} 全为零")
                
        if not has_issues:
            print("  ✓ 检查的张量数据有效，无NaN或Inf")

    def validate_weights(self):
        """验证关键权重是否正确加载"""
        if not self.c_struct.input_embd:
            raise RuntimeError("输入嵌入权重指针为空")
        if not self.c_struct.output_embd:
            raise RuntimeError("输出嵌入权重指针为空")
        if not self.c_struct.output_norm:
            raise RuntimeError("输出归一化权重指针为空")
        
        # 打印权重指针以便调试
        print("\n=== 权重结构体验证 ===")
        print(f"nlayer: {self.c_struct.nlayer}")
        
        # 修复：安全地打印枚举值
        try:
            print(f"dt_norm: {int(self.c_struct.dt_norm.value) if hasattr(self.c_struct.dt_norm, 'value') else self.c_struct.dt_norm}")
            print(f"dt_mat: {int(self.c_struct.dt_mat.value) if hasattr(self.c_struct.dt_mat, 'value') else self.c_struct.dt_mat}")
        except (ValueError, AttributeError) as e:
            print(f"dt_norm: <enum object> (cannot convert to int: {e})")
            print(f"dt_mat: <enum object> (cannot convert to int: {e})")
            
        print(f"transpose_linear_weights: {self.c_struct.transpose_linear_weights}")
        
        print(f"input_embd 指针: {hex(self.c_struct.input_embd)}")
        print(f"output_norm 指针: {hex(self.c_struct.output_norm)}")
        print(f"output_embd 指针: {hex(self.c_struct.output_embd)}")
        
        # 检查第一个图层的关键指针
        if self.c_struct.nlayer > 0:
            try:
                print(f"attn_norm[0] 指针: {hex(self.c_struct.attn_norm[0])}")
                print(f"attn_q_proj[0] 指针: {hex(self.c_struct.attn_q_proj[0])}")
                print(f"attn_k_proj[0] 指针: {hex(self.c_struct.attn_k_proj[0])}")
                print(f"attn_v_proj[0] 指针: {hex(self.c_struct.attn_v_proj[0])}")
                print(f"attn_o_proj[0] 指针: {hex(self.c_struct.attn_o_proj[0])}")
                print(f"mlp_norm[0] 指针: {hex(self.c_struct.mlp_norm[0])}")
                print(f"mlp_gate_proj[0] 指针: {hex(self.c_struct.mlp_gate_proj[0])}")
                print(f"mlp_up_proj[0] 指针: {hex(self.c_struct.mlp_up_proj[0])}")
                print(f"mlp_down_proj[0] 指针: {hex(self.c_struct.mlp_down_proj[0])}")
                
                # 打印一些权重数据，验证内容
                print("\n权重数值示例 (attn_q_proj[0]):")
                q_ptr = self.c_struct.attn_q_proj[0]
                for i, tensor in enumerate(self._tensor_refs):
                    if tensor.data_ptr() == q_ptr:
                        print(tensor.flatten()[:5].tolist())
                        break
            except Exception as e:
                print(f"指针检查失败: {e}")
        
        print("=== 验证完成 ===\n")
    
    def __getattr__(self, name):
        """代理属性访问到C结构体"""
        # 这样可以保持与原有代码的兼容性
        return getattr(self.c_struct, name)

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
            if self.tokenizer.bos_token_id is None:
                print(f"修复BOS token: 设置分词器BOS token ID = {self.meta.bos_token}")
                self.tokenizer.bos_token_id = self.meta.bos_token
                
                # 如果可能，也设置相应的文本表示
                if hasattr(self.tokenizer, '_tokenizer'):
                    try:
                        bos_text = self.tokenizer._tokenizer.id_to_token(self.meta.bos_token)
                        if bos_text:
                            self.tokenizer.bos_token = bos_text
                            print(f"  设置BOS token文本表示: '{bos_text}'")
                    except:
                        pass
    
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
        
        # 验证多设备分布
        if ndev > 1:
            self.validate_multi_device_setup(ndev)
        
        # 检查结构体布局
        self.check_struct_layout()
        
        # 验证所有指针
        self.debug_pointers_before_call()
        
        # 检查内存状态
        self.diagnose_memory_issues()
        
        # 创建C兼容的设备ID数组
        dev_ids_arr = (c_int * ndev)(*[i for i in range(ndev)])

        # 确保byref正确应用到结构体
        meta_ptr = ctypes.byref(self.meta)
        weights_ptr = ctypes.byref(self.weights.c_struct)
    
        try:
            print(f"\n==== 开始创建模型 ====")
            print(f"传递参数:")
            print(f"  meta: {ctypes.addressof(self.meta):#x}")
            print(f"  weights.c_struct: {ctypes.addressof(self.weights.c_struct):#x}")
            print(f"  device: {device}")
            print(f"  ndev: {ndev}")
            print(f"  dev_ids: {[dev_ids_arr[i] for i in range(ndev)]}")
            
            # 详细检查meta结构体值
            print("\nmeta关键字段:")
            print(f"  nlayer: {self.meta.nlayer}")
            print(f"  d: {self.meta.d}")
            print(f"  nh: {self.meta.nh}")
            print(f"  nkvh: {self.meta.nkvh}")
           
            # 修复：安全地打印dt_logits
            try:
                if hasattr(self.meta.dt_logits, 'value'):
                    print(f"  dt_logits: {int(self.meta.dt_logits.value)}")
                else:
                    print(f"  dt_logits: {self.meta.dt_logits}")
            except (ValueError, AttributeError):
                print(f"  dt_logits: <enum object>")
            
            # 详细检查weights结构体值
            print("\nweights关键字段:")
            print(f"  nlayer: {self.weights.c_struct.nlayer}")
            
            # 修复：安全地打印dt_mat和dt_norm
            try:
                if hasattr(self.weights.c_struct.dt_mat, 'value'):
                    print(f"  dt_mat: {int(self.weights.c_struct.dt_mat.value)}")
                else:
                    print(f"  dt_mat: {self.weights.c_struct.dt_mat}")
            except (ValueError, AttributeError):
                print(f"  dt_mat: <enum object>")
                
            try:
                if hasattr(self.weights.c_struct.dt_norm, 'value'):
                    print(f"  dt_norm: {int(self.weights.c_struct.dt_norm.value)}")
                else:
                    print(f"  dt_norm: {self.weights.c_struct.dt_norm}")
            except (ValueError, AttributeError):
                print(f"  dt_norm: <enum object>")
            
            # 调用C函数
            self.model_instance = create_qwen3_model(
                meta_ptr,  # 使用明确的变量
                weights_ptr,  # 使用明确的变量  
                device,
                ndev,
                dev_ids_arr  # 使用变量而非直接构造
            )
            
            # 检查返回值
            if not self.model_instance:
                raise RuntimeError("创建模型失败: 返回空指针")
                
            print(f"✓ 模型实例: {self.model_instance}")
        except Exception as e:
            print(f"✗ 模型创建失败: {e}")
            import traceback
            traceback.print_exc()
            raise

        load_end_time = time.time()
        print(f"Model creation time: {load_end_time - load_start_time:.3f}s")
        if self.model_instance is None:
            raise RuntimeError("Model instance is None after creation")

    def validate_multi_device_setup(self, ndev):
        """验证多设备设置的兼容性"""
        print(f"\n==== 多设备设置验证 ({ndev}) ====")
        # 验证模型维度是否与设备数量兼容
        if self.meta.nh % ndev != 0:
            print(f"⚠️ 注意力头数 {self.meta.nh} 不能被设备数 {ndev} 整除")
        if self.meta.nkvh % ndev != 0:
            print(f"⚠️ KV头数 {self.meta.nkvh} 不能被设备数 {ndev} 整除")
        if self.meta.di % ndev != 0:
            print(f"⚠️ 中间维度 {self.meta.di} 不能被设备数 {ndev} 整除")

    def check_struct_layout(self):
        """检查结构体内存布局与C++一致性"""
        print("\n==== 结构体内存布局验证 ====")
        
        # 打印结构体大小
        meta_size = ctypes.sizeof(self.meta)
        weights_size = ctypes.sizeof(self.weights.c_struct)
        print(f"Meta结构体大小: {meta_size}字节")
        print(f"Weights结构体大小: {weights_size}字节")
        
        # 验证结构体字段偏移量
        meta_fields = Qwen3MetaCStruct._fields_
        weights_fields = Qwen3WeightsCStruct._fields_
        
        print("\nMeta结构体字段偏移:")
        offset = 0
        for field_name, field_type in meta_fields:
            field_size = ctypes.sizeof(field_type)
            print(f"  {field_name}: 偏移={offset}, 大小={field_size}")
            offset += field_size
            
        print("\nWeights结构体字段偏移:")
        offset = 0
        for field_name, field_type in weights_fields:
            if hasattr(field_type, '_type_'):  # 指针数组
                field_size = ctypes.sizeof(field_type)
            else:
                field_size = ctypes.sizeof(field_type)
            print(f"  {field_name}: 偏移={offset}, 大小={field_size}")
            offset += field_size

    def debug_pointers_before_call(self):
        """验证所有关键指针在调用C++前的有效性"""
        print("\n==== C++调用前指针验证 ====")
        
        # 1. 基本结构体
        print(f"meta结构体地址: {ctypes.addressof(self.meta):#x}")
        print(f"weights.c_struct结构体地址: {ctypes.addressof(self.weights.c_struct):#x}")
        
        # 2. 基本字段
        print(f"nlayer: Python={self.meta.nlayer}, C结构体={self.weights.c_struct.nlayer}")
        
        # 3. 权重指针 - 检查第一个和最后一个
        print("\n关键权重指针验证:")
        nlayer = self.meta.nlayer
        
        # 输入输出权重
        print(f"input_embd: {self.weights.c_struct.input_embd:#x}")
        print(f"output_embd: {self.weights.c_struct.output_embd:#x}")
        print(f"output_norm: {self.weights.c_struct.output_norm:#x}")
        
        # 验证数组指针
        print(f"\n层权重数组指针:")
        print(f"attn_norm数组: {ctypes.addressof(self.weights.c_struct.attn_norm.contents) if self.weights.c_struct.attn_norm else 'NULL'}")
        print(f"attn_q_proj数组: {ctypes.addressof(self.weights.c_struct.attn_q_proj.contents) if self.weights.c_struct.attn_q_proj else 'NULL'}")
        
        # 验证第0层和最后一层的指针
        print(f"\n第一层(0)权重指针:")
        print(f"  attn_norm[0]: {self.weights.c_struct.attn_norm[0]:#x}")
        print(f"  attn_q_proj[0]: {self.weights.c_struct.attn_q_proj[0]:#x}")
        print(f"  attn_k_proj[0]: {self.weights.c_struct.attn_k_proj[0]:#x}")
        print(f"  attn_v_proj[0]: {self.weights.c_struct.attn_v_proj[0]:#x}")
        
        print(f"\n最后一层({nlayer-1})权重指针:")
        print(f"  attn_norm[{nlayer-1}]: {self.weights.c_struct.attn_norm[nlayer-1]:#x}")
        print(f"  attn_q_proj[{nlayer-1}]: {self.weights.c_struct.attn_q_proj[nlayer-1]:#x}")

    def diagnose_memory_issues(self):
        """诊断潜在的内存问题"""
        import gc
        import sys
        
        print("\n==== 内存诊断 ====")
        
        # 强制GC
        gc.collect()
        print(f"Python引用计数: {sys.getrefcount(self) - 1}")  # -1排除当前函数调用
        
        # 检查张量引用
        print(f"持有的张量引用数: {len(self.weights._tensor_refs)}")
        print(f"  第一个张量信息: {self.weights._tensor_refs[0].shape}, {self.weights._tensor_refs[0].dtype}")
        print(f"  最后一个张量信息: {self.weights._tensor_refs[-1].shape}, {self.weights._tensor_refs[-1].dtype}")
        
        # 检查设备内存
        try:
            if torch.cuda.is_available():
                print(f"CUDA内存分配: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
                print(f"CUDA内存缓存: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
        except:
            print("未找到CUDA或torch")

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


        # 验证输入参数
        if batch_inputs.ntok == 0:
            raise ValueError("没有tokens需要处理")
        if batch_inputs.nreq == 0:
            raise ValueError("没有请求需要处理")
        
        try:
            # 使用batch_inputs中的数组
            # print("🚀 Calling infer_qwen3_batch...")
            infer_qwen3_batch(
                self.model_instance,
                *batch_inputs.input_args(),
                output,
            )
            # print("✅ infer_qwen3_batch completed")
            
            # 验证输出token
            # for i, token in enumerate(list(output)):
            #     print(f"  Output token[{i}]: {token}")
            #     if token >= self.meta.dvoc:
            #         print(f"    ⚠ Invalid: exceeds vocab_size {self.meta.dvoc}")
            #     if token < 0:
            #         print(f"    ⚠ Invalid: negative token")
                    
        except Exception as e:
            # print(f"❌ C++ inference failed: {e}")
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
        
        # print(input_content, end="", flush=True)
        tokens = self.tokenizer.encode(input_content)

            
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
        # print(f"🔍 Model instance validation:")
        # try:
        #     model_ptr_addr = ctypes.addressof(self.model_instance.contents) if self.model_instance else 0
        #     print(f"    Model instance ptr: {hex(model_ptr_addr)}")
        # except:
        #     # 降级处理
        #     print(f"    Model instance: {self.model_instance is not None}")
            
        # if self.model_instance is None:
        #     raise RuntimeError("❌ Model instance is null before inference")


        steps = 0
        total_time = 0
        output_content = ""
        print("🚀 Starting generation:")
        for step_i in range(max_steps):
            start_time = time.time()
            output_tokens = self.batch_infer_one_round([infer_task])
            end_time = time.time()
            steps += 1
            output_token = output_tokens[0]
            try:
                output_str = self.tokenizer.decode([output_token], skip_special_tokens=False)
                print(f"{output_str}")
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
        simple_tokens = [3, 28, 1]  # BOS, simple tokens
        
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






def test_basic_functionality(model):
    """测试基本功能，排除复杂问题影响"""
    # 1. 测试非常简单的输入
    print("测试单个token输入:")
    model.generate_simple("你", 5, temperature_=0.0)
    
    # 2. 测试不同语言
    print("\n测试英文输入:")
    model.generate_simple("Hello", 5, temperature_=0.0)
    
    # 3. 测试不使用KV缓存的推理
    print("\n测试无KV缓存推理:")
    tokens = model.tokenizer.encode("测试")
    output = []
    
    for _ in range(5):
        infer_task = Qwen3InferTask(
            tokens=tokens,
            position=0,  # 始终从头开始
            temperature=0.0,
            topk=1,
            topp=1.0,
            end_tokens=[model.meta.end_token],
            max_tokens=int(model.meta.dctx),
            task_id=0
        )
        # 每次都创建新缓存
        infer_task.bind_kvcache(Qwen3KVCache(model))
        out_token = model.batch_infer_one_round([infer_task])[0]
        output.append(out_token)
        # 更新输入，但不使用缓存状态
        tokens = tokens + [out_token]
        # 清理缓存
        infer_task._kv_cache.drop()
    
    print(f"生成结果: '{model.tokenizer.decode(output)}'")

def fix_tokenizer_model_mismatch(model, model_dir_path):
    """检查并尝试修复分词器与模型之间的不匹配问题"""
    import os
    
    print("\n==== 分词器与模型匹配检查 ====")
    
    # 1. 检查词汇表大小
    model_vocab_size = model.meta.dvoc
    tokenizer_vocab_size = len(model.tokenizer.get_vocab())
    
    print(f"模型词汇表大小: {model_vocab_size}")
    print(f"分词器词汇表大小: {tokenizer_vocab_size}")
    
    if model_vocab_size != tokenizer_vocab_size:
        print(f"⚠️ 警告: 词汇表大小不匹配! 模型: {model_vocab_size}, 分词器: {tokenizer_vocab_size}")
    
    # 2. 检查特殊token
    print(f"模型BOS token ID: {model.meta.bos_token}")
    print(f"分词器BOS token ID: {model.tokenizer.bos_token_id}")
    print(f"模型EOS token ID: {model.meta.end_token}")
    print(f"分词器EOS token ID: {model.tokenizer.eos_token_id}")
    
    if model.meta.bos_token != model.tokenizer.bos_token_id or model.meta.end_token != model.tokenizer.eos_token_id:
        print(f"⚠️ 警告: 特殊token不匹配!")
    
    # 3. 检查分词器文件
    tokenizer_files = [f for f in os.listdir(model_dir_path) 
                      if f in ['tokenizer.json', 'tokenizer_config.json', 'vocab.txt', 'tokenizer.model']]
    print(f"分词器相关文件: {tokenizer_files}")
    
    # 4. 测试基本词汇的编解码
    test_words = ["你好", "Hello", "world", "测试"]
    print("\n基本词汇编解码测试:")
    for text in test_words:
        tokens = model.tokenizer.encode(text)
        decoded = model.tokenizer.decode(tokens)
        print(f"'{text}' -> {tokens} -> '{decoded}'")
    
    # 5. 检查配置文件中的词汇表大小
    try:
        import json
        with open(os.path.join(model_dir_path, "config.json"), "r") as f:
            config = json.load(f)
            config_vocab_size = config.get('vocab_size')
            print(f"配置文件中的词汇表大小: {config_vocab_size}")
            
            if config_vocab_size != model_vocab_size:
                print(f"⚠️ 警告: 配置文件词汇表大小与模型不匹配!")
    except Exception as e:
        print(f"无法读取配置文件: {e}")
    
    print("\n==== 解决方案建议 ====")
    if model_vocab_size != tokenizer_vocab_size:
        print("1. 确保使用正确的分词器文件，与模型权重配套")
        print("2. 尝试从原始Hugging Face仓库重新下载完整模型")
        print("3. 检查转换过程是否正确保留了所有分词器文件")
    
    return tokenizer_vocab_size == model_vocab_size

def verify_model_weights(model_dir_path):
    # 1. 检查权重文件的完整性
    import os
    safetensors_files = [f for f in os.listdir(model_dir_path) if f.endswith('.safetensors')]
    print(f"发现权重文件: {safetensors_files}")
    
    # 2. 验证权重文件大小是否正确
    total_size = sum(os.path.getsize(os.path.join(model_dir_path, f)) for f in safetensors_files)
    print(f"权重总大小: {total_size / (1024**3):.2f} GB")
    
    # 3. 检查配置文件
    with open(os.path.join(model_dir_path, "config.json"), "r") as f:
        import json
        config = json.load(f)
        print(f"模型类型: {config.get('model_type')}")
        print(f"配置文件中的词汇表大小: {config.get('vocab_size')}")
        # 重要: 检查这个大小是否匹配meta.dvoc

def fix_qwen3_specific_issues(model_dir_path):
    # 1. 使用正确的模型加载方式
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # 2. 验证模型配置是否正确
    import json
    with open(os.path.join(model_dir_path, "config.json"), "r") as f:
        config = json.load(f)
    
    # 3. 确认配置中的关键字段
    expected_fields = [
        "model_type", "vocab_size", "hidden_size", "num_attention_heads",
        "num_key_value_heads", "intermediate_size"
    ]
    
    for field in expected_fields:
        if field not in config:
            print(f"⚠️ 配置文件缺少关键字段: {field}")
    
    # 4. 使用相同版本的transformers库
    import transformers
    print(f"当前transformers版本: {transformers.__version__}")
    print("推荐版本: 请参考Qwen3官方文档")
def debug_pointers_before_call(self):
    """验证所有关键指针在调用C++前的有效性"""
    print("\n==== C++调用前指针验证 ====")
    
    # 1. 基本结构体
    print(f"meta结构体地址: {ctypes.addressof(self.meta):#x}")
    print(f"weights.c_struct结构体地址: {ctypes.addressof(self.weights.c_struct):#x}")
    
    # 2. 基本字段
    print(f"nlayer: Python={self.meta.nlayer}, C结构体={self.weights.c_struct.nlayer}")
    
    # 3. 权重指针 - 检查第一个和最后一个
    print("\n关键权重指针验证:")
    nlayer = self.meta.nlayer
    
    # 输入输出权重
    print(f"input_embd: {self.weights.c_struct.input_embd:#x}")
    print(f"output_embd: {self.weights.c_struct.output_embd:#x}")
    print(f"output_norm: {self.weights.c_struct.output_norm:#x}")
    
    # 验证数组指针
    print(f"\n层权重数组指针:")
    print(f"attn_norm数组: {ctypes.addressof(self.weights.c_struct.attn_norm.contents) if self.weights.c_struct.attn_norm else 'NULL'}")
    print(f"attn_q_proj数组: {ctypes.addressof(self.weights.c_struct.attn_q_proj.contents) if self.weights.c_struct.attn_q_proj else 'NULL'}")
    
    # 验证第0层和最后一层的指针
    print(f"\n第一层(0)权重指针:")
    print(f"  attn_norm[0]: {self.weights.c_struct.attn_norm[0]:#x}")
    print(f"  attn_q_proj[0]: {self.weights.c_struct.attn_q_proj[0]:#x}")
    print(f"  attn_k_proj[0]: {self.weights.c_struct.attn_k_proj[0]:#x}")
    print(f"  attn_v_proj[0]: {self.weights.c_struct.attn_v_proj[0]:#x}")
    
    print(f"\n最后一层({nlayer-1})权重指针:")
    print(f"  attn_norm[{nlayer-1}]: {self.weights.c_struct.attn_norm[nlayer-1]:#x}")
    print(f"  attn_q_proj[{nlayer-1}]: {self.weights.c_struct.attn_q_proj[nlayer-1]:#x}")
def test():
    if len(sys.argv) < 2:
        print("Usage: python qwen3.py <path/to/model_dir> [device] [n_device]")
        sys.exit(1)
        
    model_path = sys.argv[1]  # 从命令行参数获取模型路径
    device_type = DeviceType.DEVICE_TYPE_CPU
    
    if len(sys.argv) > 2:
        if sys.argv[2] == "--cpu":
            device_type = DeviceType.DEVICE_TYPE_CPU
        elif sys.argv[2] == "--nvidia":
            device_type = DeviceType.DEVICE_TYPE_NVIDIA

    ndev = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    
    print(f"✓ 使用Qwen3模型: {model_path}")
    print(f"✓ 设备: {device_type}, 设备数: {ndev}")
    
    model = QwenForCausalLM(model_path, device_type, ndev)
    
    # 诊断词汇表匹配问题
    fix_tokenizer_model_mismatch(model, model_path)

    verify_model_weights(model_path)

    fix_qwen3_specific_issues(model_path)
    
    # 诊断C++计算问题
    model.diagnose_cpp_computation()
    
    # 然后测试生成
    model.generate("山东最高的山是？", 5, topp_=0.8, topk_=50, temperature_=0.7)
    model.destroy_model_instance()

if __name__ == "__main__":
    test()