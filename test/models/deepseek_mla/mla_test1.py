import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os
import time
import ctypes
import numpy as np
import argparse
from typing import Optional, Tuple, List

# 将项目根目录添加到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

try:
    from scripts.libinfinicore_infer.deepseek_v3 import (
        DeepSeekV3MetaCStruct,
        DeepSeekV3WeightsCStruct,
        DeepSeekV3ModelCStruct,
        DeepSeekV3CacheCStruct,
        load_global_fn,
        load_layer_fn,
        load_layer_linear_fn,
        load_layer_mlp_fn,
        load_layer_expert_mlp_fn,
        DeepSeekV3WeightLoaderCStruct,
        DataType,
        DeepSeekV3Model,
    )
    import scripts.libinfinicore_infer.base as infini_base
except ImportError:
    print("Warning: Could not import InfiniLM scripts. InfiniLM tests will be skipped.")

# 基准测试常量
WARMUPS = 10
RUNS = 100

# 场景 1：小批量预填充
# 4 个请求：长度 64, 128, 256, 256。历史长度：512, 0, 0, 256
PREFILL_TESTCASES = {
    "seqlens": [64, 128, 256, 256],
    "pastlens": [512, 0, 0, 256]
}

# 场景 2：大批量解码
# 16 个请求。输入长度 1。历史长度：50*4, 100*4, 200*4, 400*4
DECODE_TESTCASES = {
    "seqlens": [1] * 16,
    "pastlens": [50]*4 + [100]*4 + [200]*4 + [400]*4
}

# -----------------------------------------------------------------------------
# 辅助函数
# -----------------------------------------------------------------------------

def load_library():
    # 优先使用本地 build_install
    local_build = os.path.join(project_root, "build_install")
    if os.path.exists(local_build):
        infini_root = local_build
    else:
        infini_root = os.environ.get("INFINI_ROOT")
    
    if not infini_root:
        possible_paths = [
            os.path.join(project_root, "build/linux/x86_64/release"),
            os.path.join(project_root, "build/macosx/x86_64/release"),
            os.path.join(project_root, "build/macosx/arm64/release")
        ]
        for p in possible_paths:
            if os.path.exists(p):
                infini_root = p
                break
    
    if not infini_root:
        print("Error: INFINI_ROOT environment variable not set and build directory not found.")
        return None
            
    lib_path = os.path.join(infini_root, "lib", "libinfinicore_infer.so")
    if not os.path.exists(lib_path):
        lib_path = os.path.join(infini_root, "lib", "libinfinicore_infer.dylib")
        
    if not os.path.exists(lib_path):
        print(f"Error: Library not found at {lib_path}")
        return None
        
    print(f"Loading library from: {lib_path}")
    lib = ctypes.CDLL(lib_path)
    DeepSeekV3Model.register_lib(lib)
    return lib

def torch_synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()

# -----------------------------------------------------------------------------
# PyTorch 参考实现（简化版，用于正确性验证和基线对比）
# -----------------------------------------------------------------------------

class DeepSeekV3Config:
    def __init__(self, model_path=None):
        self.hidden_size = 2048 # d
        self.num_heads = 16 # nh
        self.num_kv_heads = 16 # nkvh
        self.rope_dim = 64 # d_rope
        self.nope_dim = 128 # d_nope
        self.q_lora_rank = 128 # r_q
        self.kv_lora_rank = 128 # r_kv
        self.qk_head_dim = self.rope_dim + self.nope_dim # d_qk
        self.v_head_dim = 128 # d_v
        self.rms_norm_eps = 1e-6
        self.dtype = torch.bfloat16
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if model_path:
            self.load_from_json(model_path)
            
    def load_from_json(self, model_path):
        import json
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            print(f"Warning: Config file not found at {config_path}, using defaults.")
            return
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        self.hidden_size = cfg.get("hidden_size", self.hidden_size)
        self.num_heads = cfg.get("num_attention_heads", self.num_heads)
        self.num_kv_heads = cfg.get("num_key_value_heads", self.num_kv_heads)
        self.rope_dim = cfg.get("rope_dim", self.rope_dim)
        self.nope_dim = cfg.get("nope_dim", self.nope_dim)
        self.q_lora_rank = cfg.get("q_lora_rank", self.q_lora_rank)
        self.kv_lora_rank = cfg.get("kv_lora_rank", self.kv_lora_rank)
        self.v_head_dim = cfg.get("v_head_dim", self.v_head_dim)
        self.rms_norm_eps = cfg.get("rms_norm_eps", self.rms_norm_eps)
        self.qk_head_dim = self.rope_dim + self.nope_dim
        print(f"Loaded config from {config_path}")

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (x * self.weight.float()).to(dtype)

def apply_rotary_emb(x, cos, sin):
    d = x.shape[-1]
    x1 = x[..., :d//2]
    x2 = x[..., d//2:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return (x * cos) + (rotated * sin)

class DeepSeekV3MLA(nn.Module):
    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.config = config
        self.d = config.hidden_size
        self.nh = config.num_heads
        self.d_rope = config.rope_dim
        self.d_nope = config.nope_dim
        self.r_q = config.q_lora_rank
        self.r_kv = config.kv_lora_rank
        self.d_qk = config.qk_head_dim
        self.d_v = config.v_head_dim
        
        self.mla_norm = RMSNorm(self.d, config.rms_norm_eps)
        self.q_a_norm = RMSNorm(self.r_q, config.rms_norm_eps)
        self.kv_a_norm = RMSNorm(self.r_kv, config.rms_norm_eps)
        
        self.q_a_proj = nn.Linear(self.d, self.r_q, bias=False)
        self.q_b_proj = nn.Linear(self.r_q, self.nh * self.d_qk, bias=False)
        self.kv_a_proj = nn.Linear(self.d, self.r_kv + self.d_rope, bias=False)
        self.kv_b_proj = nn.Linear(self.r_kv, self.nh * (self.d_nope + self.d_v), bias=False)
        self.o_proj = nn.Linear(self.nh * self.d_v, self.d, bias=False)

    def forward(self, x, past_kv_cache=None, past_pe_cache=None, freqs_cis=None):
        batch_size, seq_len, _ = x.shape
        norm_x = self.mla_norm(x)
        
        q_a = self.q_a_proj(norm_x)
        q_a = self.q_a_norm(q_a)
        q = self.q_b_proj(q_a).view(batch_size, seq_len, self.nh, self.d_qk)
        
        q_nope = q[..., :self.d_nope]
        q_rope = q[..., self.d_nope:]
        
        if freqs_cis is not None:
            cos, sin = freqs_cis
            q_rope = apply_rotary_emb(q_rope, cos, sin)
        q = torch.cat([q_nope, q_rope], dim=-1)
        
        kv_a = self.kv_a_proj(norm_x)
        kv_pass = kv_a[..., :self.r_kv]
        k_rot = kv_a[..., self.r_kv:]
        kv_pass = self.kv_a_norm(kv_pass)
        
        if freqs_cis is not None:
            cos, sin = freqs_cis
            k_rot_expanded = k_rot.unsqueeze(2)
            k_rot = apply_rotary_emb(k_rot_expanded, cos, sin).squeeze(2)
            
        if past_kv_cache is not None:
            kv_pass = torch.cat([past_kv_cache, kv_pass], dim=1)
        if past_pe_cache is not None:
            k_rot = torch.cat([past_pe_cache, k_rot], dim=1)
            
        current_kv_cache = kv_pass
        current_pe_cache = k_rot
        total_seq_len = kv_pass.shape[1]
        
        kv_b = self.kv_b_proj(kv_pass).view(batch_size, total_seq_len, self.nh, self.d_nope + self.d_v)
        k_nope = kv_b[..., :self.d_nope]
        v = kv_b[..., self.d_nope:]
        
        k_rot_expanded = k_rot.unsqueeze(2).expand(-1, -1, self.nh, -1)
        k = torch.cat([k_nope, k_rot_expanded], dim=-1)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_qk)
        if seq_len > 1:
             mask = torch.triu(torch.ones(seq_len, total_seq_len, device=x.device), diagonal=1 + (total_seq_len - seq_len)).bool()
             scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
             
        attn_weights = F.softmax(scores.float(), dim=-1).to(v.dtype)
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.nh * self.d_v)
        output = self.o_proj(output)
        
        return output, current_kv_cache, current_pe_cache

def get_freqs_cis(seq_len, dim, device):
    freqs = torch.arange(0, dim, 2, device=device).float() / dim
    freqs = 1.0 / (10000 ** freqs)
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, freqs)
    cos = torch.cos(freqs).unsqueeze(0).unsqueeze(2)
    sin = torch.sin(freqs).unsqueeze(0).unsqueeze(2)
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    return cos, sin

# -----------------------------------------------------------------------------
# C++ 集成与权重加载
# -----------------------------------------------------------------------------

def copy_to_ptr(tensor, ptr):
    if ptr is None or tensor is None: return
    t = tensor.detach().cpu().contiguous()
    ctypes.memmove(ptr, t.data_ptr(), t.numel() * t.element_size())

class CppWeightLoader:
    def __init__(self, pytorch_model):
        self.model = pytorch_model
        
    def load_attn_norm(self, weights, w, layer_id):
        copy_to_ptr(self.model.mla_norm.weight, w)
    def load_attn_q_a_proj(self, weights, w, s, z, layer_id):
        copy_to_ptr(self.model.q_a_proj.weight.t(), w)
    def load_attn_q_a_layernorm(self, weights, w, layer_id):
        copy_to_ptr(self.model.q_a_norm.weight, w)
    def load_attn_q_b_proj(self, weights, w, s, z, layer_id):
        copy_to_ptr(self.model.q_b_proj.weight.t(), w)
    def load_attn_kv_a_proj_with_mqa(self, weights, w, s, z, layer_id):
        copy_to_ptr(self.model.kv_a_proj.weight.t(), w)
    def load_attn_kv_a_layernorm(self, weights, w, layer_id):
        copy_to_ptr(self.model.kv_a_norm.weight, w)
    def load_attn_kv_b_proj(self, weights, w, s, z, layer_id):
        weight = self.model.kv_b_proj.weight
        nh, d_nope, d_v, r_kv = self.model.nh, self.model.d_nope, self.model.d_v, self.model.r_kv
        weight = weight.view(nh, d_nope + d_v, r_kv)
        w_nope = weight[:, :d_nope, :].contiguous().reshape(-1, r_kv)
        w_v = weight[:, d_nope:, :].contiguous().reshape(-1, r_kv)
        new_weight = torch.cat([w_nope, w_v], dim=0)
        copy_to_ptr(new_weight.t(), w)
    def load_attn_o_proj(self, weights, w, s, z, layer_id):
        copy_to_ptr(self.model.o_proj.weight.t(), w)
    
    # Dummies
    def dummy_global(self, weights, w): pass
    def dummy_layer(self, weights, w, layer_id): pass
    def dummy_linear(self, weights, w, s, z, layer_id): pass
    def dummy_mlp(self, weights, w, s, z, g, d, u, layer_id): pass
    def dummy_expert(self, weights, w, s, z, g, d, u, layer_id, expert_id): pass

# -----------------------------------------------------------------------------
# 基准测试运行器
# -----------------------------------------------------------------------------

def run_prefill_benchmark(config, lib, device_str):
    print("\n=== Running Small Batch Prefill Benchmark ===")
    seqlens = PREFILL_TESTCASES["seqlens"]
    pastlens = PREFILL_TESTCASES["pastlens"]
    nreq = len(seqlens)
    
    # 1. 设置 C++ 模型
    model_wrapper = DeepSeekV3Model.__new__(DeepSeekV3Model)
    model_wrapper.lib = lib
    
    meta = DeepSeekV3MetaCStruct()
    meta.dt_logits = DataType.INFINI_DTYPE_BF16
    meta.dt_norm = DataType.INFINI_DTYPE_BF16
    meta.dt_quant_weight = DataType.INFINI_DTYPE_BF16 
    meta.dt_quant_scale = DataType.INFINI_DTYPE_F32
    meta.dt_quant_zero = DataType.INFINI_DTYPE_F32
    meta.dt_gate_weight = DataType.INFINI_DTYPE_BF16
    meta.dt_gate_bias = DataType.INFINI_DTYPE_BF16
    meta.n_sparse_layer = 0
    meta.n_dense_layer = 1 
    meta.d = config.hidden_size
    meta.nh = config.num_heads
    meta.nkvh = config.num_kv_heads
    meta.d_rope = config.rope_dim
    meta.d_nope = config.nope_dim
    meta.r_q = config.q_lora_rank
    meta.r_kv = config.kv_lora_rank
    meta.d_qk = config.qk_head_dim
    meta.d_v = config.v_head_dim
    meta.epsilon = config.rms_norm_eps
    meta.rope_theta = 10000.0 
    meta.dctx = 4096
    meta.dvoc = config.hidden_size 
    meta.di = 128; meta.di_moe = 128; meta.nexperts = 8; meta.kexperts = 2 # Dummies

    # 用于权重的 PyTorch 模型
    pt_model = DeepSeekV3MLA(config).to("cpu").to(torch.bfloat16)
    pt_model.eval()
    
    loader_cls = CppWeightLoader(pt_model)
    loader_struct = model_wrapper.create_weight_loader().contents
    c_callbacks = []
    def make_cb(func, type):
        cb = type(func); c_callbacks.append(cb); return cb

    loader_struct.load_attn_norm = make_cb(loader_cls.load_attn_norm, load_layer_fn)
    loader_struct.load_attn_q_a_proj = make_cb(loader_cls.load_attn_q_a_proj, load_layer_linear_fn)
    loader_struct.load_attn_q_a_layernorm = make_cb(loader_cls.load_attn_q_a_layernorm, load_layer_fn)
    loader_struct.load_attn_q_b_proj = make_cb(loader_cls.load_attn_q_b_proj, load_layer_linear_fn)
    loader_struct.load_attn_kv_a_proj_with_mqa = make_cb(loader_cls.load_attn_kv_a_proj_with_mqa, load_layer_linear_fn)
    loader_struct.load_attn_kv_a_layernorm = make_cb(loader_cls.load_attn_kv_a_layernorm, load_layer_fn)
    loader_struct.load_attn_kv_b_proj = make_cb(loader_cls.load_attn_kv_b_proj, load_layer_linear_fn)
    loader_struct.load_attn_o_proj = make_cb(loader_cls.load_attn_o_proj, load_layer_linear_fn)
    
    # 虚拟加载器
    def load_dummy_global(weights, w): pass
    def load_dummy_layer(weights, w, layer_id): pass
    def load_dummy_mlp(weights, w, s, z, g, d, u, layer_id): pass
    
    loader_struct.load_input_embd = make_cb(load_dummy_global, load_global_fn)
    loader_struct.load_output_norm = make_cb(load_dummy_global, load_global_fn)
    loader_struct.load_output_embd = make_cb(load_dummy_global, load_global_fn)
    loader_struct.load_mlp_norm = make_cb(load_dummy_layer, load_layer_fn)
    loader_struct.load_mlp_dense = make_cb(load_dummy_mlp, load_layer_mlp_fn)
    loader_struct.load_mlp_gate_weight = make_cb(load_dummy_layer, load_layer_fn)
    loader_struct.load_mlp_gate_bias = make_cb(load_dummy_layer, load_layer_fn)
    loader_struct.load_mlp_shared_experts = make_cb(load_dummy_mlp, load_layer_mlp_fn)
    loader_struct.load_mlp_experts = make_cb(loader_cls.dummy_expert, load_layer_expert_mlp_fn)

    dev_ids = (ctypes.c_int * 1)(0)
    weights = model_wrapper.create_weights(meta, infini_base.DeviceType.DEVICE_TYPE_NVIDIA, 1, dev_ids)
    c_model = model_wrapper.create_model(meta, weights)
    
    # 创建缓存（每个请求一个）
    caches = [model_wrapper.create_cache(c_model) for _ in range(nreq)]
    caches_arr = (ctypes.POINTER(DeepSeekV3CacheCStruct) * nreq)(*caches)

    # 准备输入
    total_tokens = sum(seqlens)
    tokens_flat = []
    for i in range(total_tokens): tokens_flat.append(i % 1000) # Dummy tokens
    tokens_arr = (ctypes.c_uint * total_tokens)(*tokens_flat)
    
    req_lens_arr = (ctypes.c_uint * nreq)(*seqlens)
    req_pos_arr = (ctypes.c_uint * nreq)(*pastlens)
    
    # 我们需要注入输入嵌入，因为我们跳过了 load_input_embd
    # 但是等等，C++ 代码从 w_in_embd 读取。
    # 对于性能测试，我们可以让它读取垃圾数据或零。
    # 但是，为了避免如果 w_in_embd 为空/null 导致的段错误，我们应该加载一些东西。
    # 在 create_weights 中，它分配内存。因为我们传递了一个什么都不做的虚拟加载器，
    # 内存已分配但未初始化。这对于性能测试是可以的。
    
    logits_out = torch.zeros(total_tokens, config.hidden_size, dtype=torch.bfloat16)
    logits_ptr = logits_out.data_ptr()

    lib.forwardBatchDeepSeekV3.argtypes = [
        ctypes.POINTER(DeepSeekV3ModelCStruct),
        ctypes.POINTER(ctypes.c_uint), ctypes.c_uint,
        ctypes.POINTER(ctypes.c_uint), ctypes.c_uint,
        ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(ctypes.POINTER(DeepSeekV3CacheCStruct)),
        ctypes.c_void_p
    ]

    # 热身
    print(f"Warming up {WARMUPS} times...")
    for _ in range(WARMUPS):
        lib.forwardBatchDeepSeekV3(c_model, tokens_arr, total_tokens, req_lens_arr, nreq, req_pos_arr, caches_arr, ctypes.c_void_p(logits_ptr))
    
    torch_synchronize(device_str)
    
    # 运行
    print(f"Running {RUNS} iterations...")
    start_time = time.time()
    for _ in range(RUNS):
        lib.forwardBatchDeepSeekV3(c_model, tokens_arr, total_tokens, req_lens_arr, nreq, req_pos_arr, caches_arr, ctypes.c_void_p(logits_ptr))
    torch_synchronize(device_str)
    end_time = time.time()
    
    avg_latency = (end_time - start_time) / RUNS * 1000
    print(f"Prefill Average Latency: {avg_latency:.4f} ms")
    
    return c_model, weights, caches # Return to keep alive

def run_decode_benchmark(config, lib, device_str):
    print("\n=== Running Large Batch Decoding Benchmark ===")
    seqlens = DECODE_TESTCASES["seqlens"]
    pastlens = DECODE_TESTCASES["pastlens"] # 初始历史长度
    nreq = len(seqlens)
    
    # 复用设置逻辑？为了简单起见，复制粘贴设置或重构。
    # 让我们假设如果我们没有销毁它们，我们可以复用相同的模型/权重。
    # 但为了干净起见，让我们重新设置。
    
    model_wrapper = DeepSeekV3Model.__new__(DeepSeekV3Model)
    model_wrapper.lib = lib
    
    meta = DeepSeekV3MetaCStruct()
    meta.dt_logits = DataType.INFINI_DTYPE_BF16
    meta.dt_norm = DataType.INFINI_DTYPE_BF16
    meta.dt_quant_weight = DataType.INFINI_DTYPE_BF16 
    meta.dt_quant_scale = DataType.INFINI_DTYPE_F32
    meta.dt_quant_zero = DataType.INFINI_DTYPE_F32
    meta.dt_gate_weight = DataType.INFINI_DTYPE_BF16
    meta.dt_gate_bias = DataType.INFINI_DTYPE_BF16
    meta.n_sparse_layer = 0
    meta.n_dense_layer = 1 
    meta.d = config.hidden_size
    meta.nh = config.num_heads
    meta.nkvh = config.num_kv_heads
    meta.d_rope = config.rope_dim
    meta.d_nope = config.nope_dim
    meta.r_q = config.q_lora_rank
    meta.r_kv = config.kv_lora_rank
    meta.d_qk = config.qk_head_dim
    meta.d_v = config.v_head_dim
    meta.epsilon = config.rms_norm_eps
    meta.rope_theta = 10000.0 
    meta.dctx = 4096
    meta.dvoc = config.hidden_size 
    meta.di = 128; meta.di_moe = 128; meta.nexperts = 8; meta.kexperts = 2

    pt_model = DeepSeekV3MLA(config).to("cpu").to(torch.bfloat16)
    pt_model.eval()
    loader_cls = CppWeightLoader(pt_model)
    loader_struct = model_wrapper.create_weight_loader().contents
    c_callbacks = []
    def make_cb(func, type): cb = type(func); c_callbacks.append(cb); return cb
    
    loader_struct.load_attn_norm = make_cb(loader_cls.load_attn_norm, load_layer_fn)
    loader_struct.load_attn_q_a_proj = make_cb(loader_cls.load_attn_q_a_proj, load_layer_linear_fn)
    loader_struct.load_attn_q_a_layernorm = make_cb(loader_cls.load_attn_q_a_layernorm, load_layer_fn)
    loader_struct.load_attn_q_b_proj = make_cb(loader_cls.load_attn_q_b_proj, load_layer_linear_fn)
    loader_struct.load_attn_kv_a_proj_with_mqa = make_cb(loader_cls.load_attn_kv_a_proj_with_mqa, load_layer_linear_fn)
    loader_struct.load_attn_kv_a_layernorm = make_cb(loader_cls.load_attn_kv_a_layernorm, load_layer_fn)
    loader_struct.load_attn_kv_b_proj = make_cb(loader_cls.load_attn_kv_b_proj, load_layer_linear_fn)
    loader_struct.load_attn_o_proj = make_cb(loader_cls.load_attn_o_proj, load_layer_linear_fn)
    
    def load_dummy_global(weights, w): pass
    def load_dummy_layer(weights, w, layer_id): pass
    def load_dummy_mlp(weights, w, s, z, g, d, u, layer_id): pass
    loader_struct.load_input_embd = make_cb(load_dummy_global, load_global_fn)
    loader_struct.load_output_norm = make_cb(load_dummy_global, load_global_fn)
    loader_struct.load_output_embd = make_cb(load_dummy_global, load_global_fn)
    loader_struct.load_mlp_norm = make_cb(load_dummy_layer, load_layer_fn)
    loader_struct.load_mlp_dense = make_cb(load_dummy_mlp, load_layer_mlp_fn)
    loader_struct.load_mlp_gate_weight = make_cb(load_dummy_layer, load_layer_fn)
    loader_struct.load_mlp_gate_bias = make_cb(load_dummy_layer, load_layer_fn)
    loader_struct.load_mlp_shared_experts = make_cb(load_dummy_mlp, load_layer_mlp_fn)
    loader_struct.load_mlp_experts = make_cb(loader_cls.dummy_expert, load_layer_expert_mlp_fn)

    dev_ids = (ctypes.c_int * 1)(0)
    weights = model_wrapper.create_weights(meta, infini_base.DeviceType.DEVICE_TYPE_NVIDIA, 1, dev_ids)
    c_model = model_wrapper.create_model(meta, weights)
    caches = [model_wrapper.create_cache(c_model) for _ in range(nreq)]
    caches_arr = (ctypes.POINTER(DeepSeekV3CacheCStruct) * nreq)(*caches)

    lib.forwardBatchDeepSeekV3.argtypes = [
        ctypes.POINTER(DeepSeekV3ModelCStruct),
        ctypes.POINTER(ctypes.c_uint), ctypes.c_uint,
        ctypes.POINTER(ctypes.c_uint), ctypes.c_uint,
        ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(ctypes.POINTER(DeepSeekV3CacheCStruct)),
        ctypes.c_void_p
    ]

    # 解码循环
    # 100 步。每一步，所有请求的输入长度均为 1。
    # 历史长度每步增加 1。
    
    current_pastlens = list(pastlens)
    total_tokens_generated = nreq * RUNS
    
    # 准备固定的输入缓冲区（因为我们在性能测试中不关心实际值）
    # 输入 token：16 个 token
    tokens_flat = [0] * nreq
    tokens_arr = (ctypes.c_uint * nreq)(*tokens_flat)
    req_lens_arr = (ctypes.c_uint * nreq)(*[1]*nreq)
    
    logits_out = torch.zeros(nreq, config.hidden_size, dtype=torch.bfloat16)
    logits_ptr = logits_out.data_ptr()

    print(f"Running {RUNS} decoding steps...")
    start_time = time.time()
    
    for step in range(RUNS):
        # 更新 req_pos
        req_pos_arr = (ctypes.c_uint * nreq)(*current_pastlens)
        
        lib.forwardBatchDeepSeekV3(c_model, tokens_arr, nreq, req_lens_arr, nreq, req_pos_arr, caches_arr, ctypes.c_void_p(logits_ptr))
        
        # 增加历史长度
        for i in range(nreq):
            current_pastlens[i] += 1
            
    torch_synchronize(device_str)
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_latency = total_time / total_tokens_generated * 1000
    print(f"Total Time: {total_time:.4f} s")
    print(f"Decode Average Latency per Token: {avg_latency:.4f} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    config = DeepSeekV3Config(args.model_path)
    lib = load_library()
    
    if lib:
        run_prefill_benchmark(config, lib, args.device)
        run_decode_benchmark(config, lib, args.device)
    else:
        print("Skipping benchmarks due to missing library.")