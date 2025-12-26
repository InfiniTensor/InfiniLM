import os
import time
import math
import sys
import safetensors
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig

# ================================================================= #
#  基础组件手动实现
# ================================================================= #

class ManualRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)

# [预计算] 标准实数 Cos/Sin 生成
def precompute_freqs_cos_sin(dim: int, end: int, theta: float = 10000.0, device="cpu"):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    t = torch.arange(end, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # [seq_len, dim/2]
    
    # 注意：这里我们返回原始的 [S, D/2] 形状，或者为了兼容 interleaved 计算重复一遍
    # 为了适配下面的 interleaved 逻辑（需要 cos[..., 0] 和 cos[..., 1] 相同），
    # 我们生成 [S, D] 格式，其中偶数位和奇数位相同。
    # 也就是 [freq0, freq0, freq1, freq1, ...]
    # 不过 PyTorch 的 repeat_interleave 比较方便
    
    freqs_expanded = freqs.repeat_interleave(2, dim=-1) # [S, D]
    cos = freqs_expanded.cos()
    sin = freqs_expanded.sin()
    return cos, sin

# [关键修改] 实现 DeepSeek/HF 风格的“交错式”旋转
# 对应 HF 的 apply_rotary_pos_emb 中的复数乘法逻辑
def apply_rotary_emb_interleaved(x, cos, sin):
    """
    x: [Batch, Num_Heads, Seq_Len, Head_Dim]
    cos, sin: [Seq_Len, Head_Dim]
    """
    # 1. 调整 cos/sin 形状以支持广播 [1, 1, S, D]
    cos = cos.view(1, 1, x.shape[-2], x.shape[-1]).to(x.dtype)
    sin = sin.view(1, 1, x.shape[-2], x.shape[-1]).to(x.dtype)

    # 2. 将 x 重塑为 [..., D/2, 2] 以便分离相邻的实部(0)和虚部(1)
    # 假设 Head_Dim 是 64，变成了 32 对
    x_reshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    x_real = x_reshaped[..., 0]
    x_imag = x_reshaped[..., 1]
    
    # 3. 同样处理 cos/sin
    # 因为我们在 precompute 里做了 repeat_interleave，所以 cos[..., 0] == cos[..., 1]
    # 这里取其中一个即可用于旋转公式
    cos_reshaped = cos.float().reshape(*cos.shape[:-1], -1, 2)
    sin_reshaped = sin.float().reshape(*sin.shape[:-1], -1, 2)
    
    cos_val = cos_reshaped[..., 0]
    sin_val = sin_reshaped[..., 0]

    # 4. 执行旋转公式 (Complex Multiplication)
    # (a + ib) * (cos + isin) = (a*cos - b*sin) + i(a*sin + b*cos)
    out_real = x_real * cos_val - x_imag * sin_val
    out_imag = x_real * sin_val + x_imag * cos_val
    
    # 5. 拼接回 [..., D/2, 2] 并 flatten 到 [..., D]
    out = torch.stack([out_real, out_imag], dim=-1).flatten(-2)
    
    return out.to(x.dtype)

# ================================================================= #
#  手动实现 DeepSeek V2/V3 MLA 核心逻辑
# ================================================================= #

class ManualDeepseekV2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        
        self.qk_nope_head_dim = config.qk_nope_head_dim # 128
        self.qk_rope_head_dim = config.qk_rope_head_dim # 64
        self.v_head_dim = config.v_head_dim             # 128
        
        # 强制 bias=False
        self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
        self.q_a_layernorm = ManualRMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim), bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_a_layernorm = ManualRMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(self.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False)

        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False)
        
        self.softmax_scale = 1.0 / math.sqrt(self.qk_nope_head_dim + self.qk_rope_head_dim)

    def forward(self, hidden_states, rope_cos_sin, past_key_values=None):
        bsz, q_len, _ = hidden_states.size()
        cos, sin = rope_cos_sin

        # 1. Q 处理
        q = self.q_a_proj(hidden_states)
        q = self.q_a_layernorm(q)
        q = self.q_b_proj(q)
        q = q.view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.qk_rope_head_dim)
        
        # Split -> Transpose
        q_nope, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_nope = q_nope.transpose(1, 2)
        q_rope = q_rope.transpose(1, 2) # [B, H, S, 64]

        # 2. KV 处理
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_rope = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        
        # Expand k_rope (MLA 中 RoPE 部分是共享的，但在计算 Attention Score 时需要广播到多头)
        k_rope = k_rope.view(bsz, q_len, 1, self.qk_rope_head_dim).expand(-1, -1, self.num_heads, -1)
        k_rope = k_rope.transpose(1, 2)

        norm_kv = self.kv_a_layernorm(compressed_kv)
        kv = self.kv_b_proj(norm_kv)
        kv = kv.view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, value = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_nope = k_nope.transpose(1, 2)
        value = value.transpose(1, 2)

        # 3. RoPE (使用交错式旋转，对齐 HF)
        q_rope = apply_rotary_emb_interleaved(q_rope, cos, sin)
        k_rope = apply_rotary_emb_interleaved(k_rope, cos, sin)

        # 4. Cache Update
        if past_key_values is not None:
            if len(past_key_values) == 0:
                past_key_values.extend([k_nope, k_rope, value])
            else:
                k_nope = torch.cat([past_key_values[0], k_nope], dim=2)
                k_rope = torch.cat([past_key_values[1], k_rope], dim=2)
                value  = torch.cat([past_key_values[2], value],  dim=2)
                past_key_values[0] = k_nope
                past_key_values[1] = k_rope
                past_key_values[2] = value

        # 5. Attention
        query = torch.cat([q_nope, q_rope], dim=-1) 
        key   = torch.cat([k_nope, k_rope], dim=-1) 
        
        attn_weights = torch.matmul(query, key.transpose(2, 3)) * self.softmax_scale
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        
        attn_output = torch.matmul(attn_weights, value) 

        # 6. Output Proj
        attn_output = attn_output.transpose(1, 2).contiguous() 
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        output = self.o_proj(attn_output)
        
        return output

# ================================================================= #
#  Runner Logic (Standalone Test Support)
# ================================================================= #
WARMUPS = 10
RUNS = 100
PREFILL_TESTCASES = {"seqlens": [64, 128, 256, 256], "pastlens": [512, 0, 0, 256]}
DECODE_TESTCASES = {
    "seqlens": [1 for _ in range(16)],
    "pastlens": [50 for _ in range(4)] + [100 for _ in range(4)] + [200 for _ in range(4)] + [400 for _ in range(4)],
}

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--nvidia", action="store_true")
    return parser.parse_args()

def torch_synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()

def create_manual_mla_torch(dir_path, *, device, dtype=torch.bfloat16):
    config = AutoConfig.from_pretrained(dir_path)
    print(f"Initializing Manual DeepseekV2Attention (Torch)...")
    model = ManualDeepseekV2Attention(config).to(device=device, dtype=dtype)
    
    # 这里的加载逻辑主要用于独立运行时的测试，check 脚本会覆盖这里
    print(f"Loading weights from {dir_path}...")
    tensors = {}
    for fname in sorted(os.listdir(dir_path)):
        if not fname.endswith(".safetensors"): continue
        fpath = os.path.join(dir_path, fname)
        with safetensors.safe_open(fpath, framework="pt") as f:
            for key in f.keys():
                if "self_attn." in key:
                    new_key = key.split("self_attn.")[1]
                    tensors[new_key] = f.get_tensor(key)
    model.load_state_dict(tensors, strict=False)
    model.eval()
    return model

def generate_mla_inputs_torch(model, testcase, device, dtype):
    config = model.config
    hidden_size = config.hidden_size
    rope_theta = getattr(config, "rope_theta", 10000.0)
    
    # 预计算 Cos/Sin (足够长)
    max_len = 32768
    full_cos, full_sin = precompute_freqs_cos_sin(config.qk_rope_head_dim, max_len, theta=rope_theta, device=device)
    
    bs = 1
    req_list = []
    for seq_lens, past_lens in zip(testcase["seqlens"], testcase["pastlens"]):
        hidden_states = torch.rand((bs, seq_lens, hidden_size), device=device, dtype=dtype)
        past_key_values = [] 
        req = {
            "hidden_states": hidden_states,
            "past_key_values": past_key_values,
            "past_len": past_lens,
            "rope_cos": full_cos, # 传递全量表
            "rope_sin": full_sin
        }
        req_list.append(req)
    return req_list

def benchmark_mla_prefill_torch(model, test_cases, device, dtype):
    req_list = generate_mla_inputs_torch(model, test_cases, device, dtype)
    
    # Fill Cache
    for req in req_list:
        if req["past_len"] > 0:
            dummy_len = req["past_len"]
            dummy_input = torch.zeros((1, dummy_len, model.hidden_size), device=device, dtype=dtype)
            cos = req["rope_cos"][:dummy_len]
            sin = req["rope_sin"][:dummy_len]
            model(dummy_input, (cos, sin), req["past_key_values"])

    # Warmup
    for _ in range(WARMUPS):
        for req in req_list:
            start = len(req["past_key_values"][0][0][0]) if req["past_key_values"] else 0
            end = start + req["hidden_states"].shape[1]
            cos = req["rope_cos"][start:end]
            sin = req["rope_sin"][start:end]
            model(req["hidden_states"], (cos, sin), past_key_values=None)

    torch_synchronize(device)
    start_time = time.time()

    for _ in range(RUNS):
        for i, req in enumerate(req_list):
            req["past_key_values"] = []
            if req["past_len"] > 0:
                 # 模拟已有 Cache
                 k_nope = torch.randn(1, model.num_heads, req["past_len"], 128, device=device, dtype=dtype)
                 k_rope = torch.randn(1, model.num_heads, req["past_len"], 64, device=device, dtype=dtype)
                 val    = torch.randn(1, model.num_heads, req["past_len"], 128, device=device, dtype=dtype)
                 req["past_key_values"] = [k_nope, k_rope, val]

            start = req["past_len"]
            end = start + req["hidden_states"].shape[1]
            cos = req["rope_cos"][start:end]
            sin = req["rope_sin"][start:end]
            
            model(req["hidden_states"], (cos, sin), req["past_key_values"])

    torch_synchronize(device)
    end_time = time.time()
    ttft = (end_time - start_time) * 1000 / RUNS
    print(f"\t WARMUPS={WARMUPS} RUNS={RUNS}, MLA Manual Torch, average TTFT: {round(ttft, 2)} ms\n")

def benchmark_mla_decode_torch(model, test_cases, device, dtype):
    req_list = generate_mla_inputs_torch(model, test_cases, device, dtype)
    
    print("Initializing KV Cache for Decode...")
    for req in req_list:
        if req["past_len"] > 0:
             k_nope = torch.randn(1, model.num_heads, req["past_len"], 128, device=device, dtype=dtype)
             k_rope = torch.randn(1, model.num_heads, req["past_len"], 64, device=device, dtype=dtype)
             val    = torch.randn(1, model.num_heads, req["past_len"], 128, device=device, dtype=dtype)
             req["past_key_values"] = [k_nope, k_rope, val]

    torch_synchronize(device)
    # Warmup
    for req in req_list:
        for _ in range(WARMUPS):
            start = req["past_len"]
            cos = req["rope_cos"][start : start+1]
            sin = req["rope_sin"][start : start+1]
            model(req["hidden_states"], (cos, sin), None)

    torch_synchronize(device)
    start_time = time.time()

    for i, req in enumerate(req_list):
        for _ in range(RUNS):
            curr_len = req["past_key_values"][0].shape[2]
            cos = req["rope_cos"][curr_len : curr_len+1]
            sin = req["rope_sin"][curr_len : curr_len+1]
            
            output = model(req["hidden_states"], (cos, sin), req["past_key_values"])
            req["hidden_states"] = output

    torch_synchronize(device)
    end_time = time.time()
    throughput = (RUNS * len(req_list)) / (end_time - start_time)
    print(f"\t WARMUPS={WARMUPS} RUNS={RUNS}, MLA Manual Torch, average throughput: {round(throughput, 2)} tok/s \n")

if __name__ == "__main__":
    args = get_args()
    device = torch.device("cuda" if args.nvidia else "cpu")
    dtype = torch.bfloat16
    model = create_manual_mla_torch(args.model_path, device=device, dtype=dtype)
    
    print("\n" + "*" * 130)
    print("Test DeepSeek-R1 MLA (Manual PyTorch Reference)")
    print("*" * 130)
    print(f"Test Case PREFILL: {PREFILL_TESTCASES}")
    benchmark_mla_prefill_torch(model, PREFILL_TESTCASES, device, dtype)
    print(f"\nTest Case DECODE: {DECODE_TESTCASES}")
    benchmark_mla_decode_torch(model, DECODE_TESTCASES, device, dtype)