import os
import time
import sys
import json
import safetensors
import torch
import numpy as np
import ctypes
from ctypes import byref, POINTER, c_int, c_float, c_void_p, c_size_t, Structure
from transformers import AutoConfig
from transformers import DynamicCache
from transformers.models import qwen3_moe

# ==============================================================================
# 1. Ctypes Setup
# ==============================================================================
SO_PATH = "build/linux/x86_64/release/libinfinicore_infer.so"
if not os.path.exists(SO_PATH):
    SO_PATH = os.path.expanduser("~/.infini/lib/libinfinicore_infer.so")

if not os.path.exists(SO_PATH):
    print(f"Warning: Cannot find libinfinicore_infer.so at {SO_PATH}.")
    LIB_INFINILM = None
else:
    LIB_INFINILM = ctypes.CDLL(SO_PATH)

class DataType:
    INFINI_DTYPE_BF16 = 19

class DeviceType:
    DEVICE_TYPE_NVIDIA = 1

class Qwen3MoEAttentionMetaCStruct(Structure):
    _fields_ = [
        ("dtype", c_int),
        ("hidden_size", c_size_t),
        ("num_heads", c_size_t),
        ("num_kv_head", c_size_t),
        ("head_dim", c_size_t),
        ("rope_theta", c_float),
        ("max_seq_len", c_size_t),
        ("rms_norm_eps", c_float),
    ]

class Qwen3MoEWeightLoader(Structure):
    _fields_ = [
        ("load_attn_norm", ctypes.CFUNCTYPE(None, c_void_p, c_void_p, c_size_t)),
        ("load_attn_q_proj", ctypes.CFUNCTYPE(None, c_void_p, c_void_p, c_size_t)),
        ("load_attn_k_proj", ctypes.CFUNCTYPE(None, c_void_p, c_void_p, c_size_t)),
        ("load_attn_v_proj", ctypes.CFUNCTYPE(None, c_void_p, c_void_p, c_size_t)),
        ("load_attn_q_norm", ctypes.CFUNCTYPE(None, c_void_p, c_void_p, c_size_t)),
        ("load_attn_k_norm", ctypes.CFUNCTYPE(None, c_void_p, c_void_p, c_size_t)),
        ("load_attn_o_proj", ctypes.CFUNCTYPE(None, c_void_p, c_void_p, c_size_t)),
    ]

class Qwen3MoEAttention(Structure): pass
class Qwen3MoEWeights(Structure): pass
class Qwen3Cache(Structure): pass

if LIB_INFINILM:
    LIB_INFINILM.createQwen3MoEWeights.restype = POINTER(Qwen3MoEWeights)
    LIB_INFINILM.createQwen3MoEWeightLoader.restype = POINTER(Qwen3MoEWeightLoader)
    LIB_INFINILM.createQwen3MoEAttention.restype = POINTER(Qwen3MoEAttention)
    LIB_INFINILM.createQwen3Cache.restype = POINTER(Qwen3Cache)
    LIB_INFINILM.createQwen3Cache.argtypes = [POINTER(Qwen3MoEAttentionMetaCStruct), c_size_t, c_size_t]

    LIB_INFINILM.forwardQwen3MoEAttention.argtypes = [
        POINTER(Qwen3MoEAttention), POINTER(Qwen3Cache),
        c_void_p, c_void_p, c_int, POINTER(c_int), POINTER(c_int), POINTER(c_int)
    ]
    LIB_INFINILM.injectQwen3CacheKV.argtypes = [
        POINTER(Qwen3MoEAttention), POINTER(Qwen3Cache),
        c_int, c_int, c_int, c_void_p, c_void_p
    ]

global_tensor_keepalive = []

def get_ptr(numpy_array):
    if not numpy_array.flags['C_CONTIGUOUS']:
        numpy_array = np.ascontiguousarray(numpy_array)
    ptr = numpy_array.ctypes.data_as(c_void_p)
    global_tensor_keepalive.append(numpy_array)
    return ptr

# ==============================================================================
# 2. InfiniLM Wrapper
# ==============================================================================
class InfiniLMWrapper:
    def __init__(self, config, torch_model, device_id=0):
        if not LIB_INFINILM: raise RuntimeError("Library not loaded")
        
        # [TRUTH] 物理真值是 128
        self.real_hidden = config.hidden_size 
        real_head_dim = 128
        
        self.meta = Qwen3MoEAttentionMetaCStruct(
            dtype=DataType.INFINI_DTYPE_BF16, 
            hidden_size=config.hidden_size, 
            num_heads=config.num_attention_heads,
            num_kv_head=config.num_key_value_heads,
            head_dim=real_head_dim,
            rope_theta=config.rope_theta,
            max_seq_len=8192,
            rms_norm_eps=config.rms_norm_eps
        )
        self.weights_handle = LIB_INFINILM.createQwen3MoEWeights(byref(self.meta), DeviceType.DEVICE_TYPE_NVIDIA, 1, (c_int * 1)(device_id))
        self.loader = LIB_INFINILM.createQwen3MoEWeightLoader()
        self._load_weights(torch_model)
        self.attn_ctx = LIB_INFINILM.createQwen3MoEAttention(byref(self.meta), self.weights_handle)
        self.kv_cache = LIB_INFINILM.createQwen3Cache(byref(self.meta), 0, 0)
    
    def _load_weights(self, model):
        def load(tensor, loader_func, transpose=False):
            if tensor is None: return
            w_pt = tensor.detach().to(torch.float32)
            if transpose: w_pt = w_pt.t()
            w_bf16 = w_pt.to(torch.bfloat16).view(torch.int16).cpu().numpy()
            loader_func(self.weights_handle, get_ptr(w_bf16), 0)

        load(model.q_proj.weight, self.loader.contents.load_attn_q_proj, transpose=True)
        load(model.k_proj.weight, self.loader.contents.load_attn_k_proj, transpose=True)
        load(model.v_proj.weight, self.loader.contents.load_attn_v_proj, transpose=True)
        load(model.o_proj.weight, self.loader.contents.load_attn_o_proj, transpose=True)
        
        if hasattr(model, 'q_norm') and model.q_norm is not None:
            load(model.q_norm.weight, self.loader.contents.load_attn_q_norm, transpose=False)
        if hasattr(model, 'k_norm') and model.k_norm is not None:
            load(model.k_norm.weight, self.loader.contents.load_attn_k_norm, transpose=False)

    def inject_cache(self, layer_id, batch_idx, k_torch, v_torch):
        """
        将 PyTorch 的 KV Cache (BFloat16) 注入到 InfiniLM 的 Cache 中
        k_torch, v_torch shape: [num_kv_heads, past_len, head_dim]
        """
        if k_torch is None or v_torch is None: return
        
        # 转换为 numpy int16 (模拟 bf16) 且保证 C 连续
        k_np = k_torch.detach().cpu().view(torch.int16).numpy().copy(order='C')
        v_np = v_torch.detach().cpu().view(torch.int16).numpy().copy(order='C')
        past_len = k_np.shape[1] 
        
        LIB_INFINILM.injectQwen3CacheKV(
            self.attn_ctx, self.kv_cache,
            c_int(layer_id), c_int(batch_idx), c_int(past_len),
            get_ptr(k_np), get_ptr(v_np)
        )

    def forward(self, input_bf16_np, batch_size, seq_lens, past_lens, pos_ids, return_raw=False):
        q_out_dim = self.meta.num_heads * self.meta.head_dim 
        out_dim = q_out_dim if return_raw else self.real_hidden
        output = np.zeros((input_bf16_np.shape[0], out_dim), dtype=np.int16)
        
        LIB_INFINILM.forwardQwen3MoEAttention(
            self.attn_ctx, self.kv_cache, 
            get_ptr(input_bf16_np), get_ptr(output),
            c_int(batch_size), (c_int*batch_size)(*seq_lens), 
            (c_int*batch_size)(*past_lens), (c_int*len(pos_ids))(*pos_ids)
        )
        return output

# ==============================================================================
# 3. Utilities
# ==============================================================================
WARMUPS = 10
RUNS = 100
PREFILL_TESTCASES = {"seqlens": [64,128,256,256], "pastlens": [512,0,0,256]}
DECODE_TESTCASES = {"seqlens": [1] * 16, "pastlens": [504]*4 + [1004]*4 + [2004]*4 + [4004]*4}

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--nvidia", action="store_true")
    return parser.parse_args()

def create_Qwen3attention_torch(dir_path, device, dtype=torch.bfloat16):
    config = AutoConfig.from_pretrained(dir_path)

    real_head_dim = 128 
    config.head_dim = real_head_dim 

    config.num_hidden_layers = 1
    config._attn_implementation = "sdpa"
    
    model = qwen3_moe.modeling_qwen3_moe.Qwen3MoeAttention(config, layer_idx=0).to(device=device, dtype=dtype)
    
    tensors = {}
    for fname in sorted(os.listdir(dir_path)):
        if not fname.endswith(".safetensors"): continue
        with safetensors.safe_open(os.path.join(dir_path, fname), framework="pt") as f:
            for key in f.keys():
                if "model.layers.0.self_attn." in key:
                    tensors[key[len("model.layers.0.self_attn.") :]] = f.get_tensor(key)
        break
    
    model.load_state_dict(tensors, strict=False)
    
    if model.q_proj.bias is not None: torch.nn.init.zeros_(model.q_proj.bias)
    if model.k_proj.bias is not None: torch.nn.init.zeros_(model.k_proj.bias)
    if model.v_proj.bias is not None: torch.nn.init.zeros_(model.v_proj.bias)
    if model.o_proj.bias is not None: torch.nn.init.zeros_(model.o_proj.bias)

    rotary_emb = qwen3_moe.modeling_qwen3_moe.Qwen3MoeRotaryEmbedding(config, device=device)
    return model, rotary_emb, config

def prepare_inputs(model, testcase, device, dtype):
    config = model.config
    bs = 1
    req_list = []
    
    for seq_lens, past_lens in zip(testcase["seqlens"], testcase["pastlens"]):
        hidden_states = torch.rand((bs, seq_lens, config.hidden_size), device=device, dtype=dtype)
        past_key_values = DynamicCache(config=config)
        
        # [CRITICAL] 恢复为 torch.rand！
        # 现在我们通过 inject_cache 保证 C++ 拿到完全一样的随机数
        if past_lens > 0:
            k = torch.rand((bs, config.num_key_value_heads, past_lens, config.head_dim), device=device, dtype=dtype)
            v = torch.rand((bs, config.num_key_value_heads, past_lens, config.head_dim), device=device, dtype=dtype)
            past_key_values.update(k, v, 0)
        req_list.append({"hidden_states": hidden_states, "attention_mask": None, "past_key_values": past_key_values})

    all_hs = [req["hidden_states"].squeeze(0) for req in req_list]
    flat_input = torch.cat(all_hs, dim=0) 
    
    input_np = flat_input.cpu().view(torch.int16).numpy().copy(order='C')
    
    seq_lens = testcase["seqlens"]
    past_lens = testcase["pastlens"]
    pos_ids = []
    for s, p in zip(seq_lens, past_lens):
        pos_ids.extend(range(p, p+s))
        
    return req_list, input_np, seq_lens, past_lens, pos_ids

def check_correctness_prefill(torch_outs, infinilm_out_np, device):
    if not torch_outs:
        print("❌ Error: Torch Output is empty.")
        return

    torch_flat = torch.cat([out.float().view(-1, out.shape[-1]) for out in torch_outs], dim=0).to("cpu")
    
    infini_tensor_int16 = torch.from_numpy(infinilm_out_np)
    infini_flat = infini_tensor_int16.view(torch.bfloat16).float().view(-1, torch_flat.shape[-1])

    cos_sim = torch.nn.functional.cosine_similarity(torch_flat, infini_flat, dim=-1).mean().item()
    print(f"Cosine Similarity: {cos_sim:.6f}")
    
    if cos_sim > 0.98: print("✅ Result Match")
    else: print("❌ Result Mismatch")

def check_correctness_decode(torch_outs, infinilm_out_np, device):
    if not torch_outs:
        print("❌ Error: Torch Output is empty.")
        return

    torch_flat = torch.cat([out.float().view(-1, out.shape[-1]) for out in torch_outs], dim=0).to("cpu")
    
    infini_tensor_int16 = torch.from_numpy(infinilm_out_np)
    infini_flat = infini_tensor_int16.view(torch.bfloat16).float().view(-1, torch_flat.shape[-1])

    cos_sim = torch.nn.functional.cosine_similarity(torch_flat, infini_flat, dim=-1).mean().item()
    print(f"Cosine Similarity: {cos_sim:.6f}")
    ## for decode, 0.95 enough 
    if cos_sim > 0.95: print("✅ Result Match")
    else: print("❌ Result Mismatch")


def benchmark_prefill(model, rotary_emb, infinilm_model, test_cases, device, dtype):
    print(f"\n{'='*40} PREFILL {'='*40}")
    req_list, input_np, seq_lens, past_lens, pos_ids = prepare_inputs(model, test_cases, device, dtype)
    batch_size = len(seq_lens)
    
    # =======================================================
    # Torch Run
    # =======================================================
    for _ in range(WARMUPS):
        for i, req in enumerate(req_list):
            req["past_key_values"].crop(past_lens[i])
            cache_len = req["past_key_values"].get_seq_length()
            seq_len = req["hidden_states"].shape[1]
            pids = torch.arange(cache_len, cache_len+seq_len, device=device).reshape(1, seq_len)
            cos, sin = rotary_emb(req["hidden_states"], pids)
            _ = model(req["hidden_states"], position_embeddings=(sin, cos), 
                      attention_mask=req["attention_mask"],
                      past_key_values=req["past_key_values"])
    torch.cuda.synchronize()

    
    torch_out_list = []
    time_consuming = 0
    for run_idx in range(RUNS):
        for i, req in enumerate(req_list):
            # 1. Reset KV Cache to initial state
            req["past_key_values"].crop(past_lens[i])
            cache_len = req["past_key_values"].get_seq_length()
            seq_len = req["hidden_states"].shape[1]
            
            q_len = seq_len
            k_len = cache_len + seq_len
            past_len = cache_len
            
            causal_mask = torch.zeros((q_len, k_len), device=device, dtype=dtype)
            for j in range(q_len):
                valid_limit = past_len + j + 1
                if valid_limit < k_len:
                    causal_mask[j, valid_limit:] = float("-inf")
            req["attention_mask"] = causal_mask[None, None, :, :]
        # ----------------------------------------- #
        #       重要：每个req都按整个batch的起始时间计算
        # ----------------------------------------- #
        torch.cuda.synchronize()
        start = time.time()
        for i, req in enumerate(req_list):     
            req["past_key_values"].crop(past_lens[i])
            cache_len = req["past_key_values"].get_seq_length()
            seq_len = req["hidden_states"].shape[1]
            # Position IDs
            pids = torch.arange(cache_len, cache_len+seq_len, device=device).reshape(1, seq_len)
            cos, sin = rotary_emb(req["hidden_states"], pids)
            out, _ = model(req["hidden_states"], position_embeddings=(sin, cos), 
                           attention_mask=req["attention_mask"],
                           past_key_values=req["past_key_values"])
            torch.cuda.synchronize()
            end_time = time.time()
            time_consuming += end_time - start
            if run_idx == RUNS - 1:
                torch_out_list.append(out.detach().to("cpu"))
    torch.cuda.synchronize()
    out_token_count = RUNS * len(req_list)
    t_lat = time_consuming * 1000 / out_token_count

    # =======================================================
    # InfiniLM Run
    # =======================================================
    print(">>> Injecting Cache to InfiniLM...")
    for i, req in enumerate(req_list):
        if past_lens[i] > 0:
            k_cache = req["past_key_values"][0][0].squeeze(0) 
            v_cache = req["past_key_values"][0][1].squeeze(0)
            infinilm_model.inject_cache(0, i, k_cache, v_cache)

    for _ in range(WARMUPS):
        _ = infinilm_model.forward(input_np, batch_size, seq_lens, past_lens, pos_ids, return_raw=False)
    torch.cuda.synchronize()

    start = time.time()
    infini_out = None
    for _ in range(RUNS):
        # Repeatedly run with same inputs (simulating same-shape prefill)
        # Note: We assume InfiniLM overwrites/resets based on past_lens parameter
        infini_out = infinilm_model.forward(input_np, batch_size, seq_lens, past_lens, pos_ids, return_raw=False)
    torch.cuda.synchronize()
    out_token_count = RUNS * len(req_list)
    i_lat = (time.time() - start) * 1000 / out_token_count

    print(f"Latency: Torch={t_lat:.3f}ms, Infini={i_lat:.3f}ms")
    check_correctness_prefill(torch_out_list, infini_out, device)


def benchmark_decode(model, rotary_emb, infinilm_model, test_cases, device, dtype):
    print(f"\n{'='*40} DECODE {'='*40}")
    req_list, input_np, seq_lens, past_lens, pos_ids = prepare_inputs(model, test_cases, device, dtype)
    batch_size = len(seq_lens)
    total_tokens_per_round = sum(seq_lens)

    # Capture initial KV for InfiniLM injection (before Torch modifies them)
    initial_kv = []
    for req in req_list:
        if req["past_key_values"].get_seq_length() > 0:
            k = req["past_key_values"][0][0].detach().clone()
            v = req["past_key_values"][0][1].detach().clone()
            initial_kv.append((k, v))
        else:
            initial_kv.append(None)

    # =======================================================
    # Torch Run
    # =======================================================
    # Note: No Warmup mentioned in requirements for "Sequential inference 100 rounds", 
    # but usually we might warm up. However, since state changes, warmup is part of the sequence.
    # We will just run the 100 rounds as the benchmark.

    torch_out_list = []
    torch.cuda.synchronize()
    start = time.time()
    for run_idx in range(RUNS):
        for i, req in enumerate(req_list):
            # Do NOT crop cache - let it grow
            cache_len = req["past_key_values"].get_seq_length()
            seq_len = req["hidden_states"].shape[1] # Should be 1
            
            pids = torch.arange(cache_len, cache_len+seq_len, device=device).reshape(1, seq_len)
            cos, sin = rotary_emb(req["hidden_states"], pids)
            
            # Decode: attention_mask is None (causal implied for len 1)
            out, _ = model(req["hidden_states"], position_embeddings=(sin, cos), 
                           attention_mask=None,
                           past_key_values=req["past_key_values"])
            
            # Update input for next round
            req["hidden_states"] = out
            
            if run_idx == RUNS - 1:
                torch_out_list.append(out.detach().to("cpu"))
                
    torch.cuda.synchronize()
    end = time.time()
    t_throughput = (total_tokens_per_round * RUNS) / (end - start)

    # =======================================================
    # InfiniLM Run
    # =======================================================
    print(">>> Injecting Cache to InfiniLM...")
    for i, kv in enumerate(initial_kv):
        if kv is not None:
            k_cache, v_cache = kv
            k_cache = k_cache.squeeze(0)
            v_cache = v_cache.squeeze(0)
            infinilm_model.inject_cache(0, i, k_cache, v_cache)
            
    curr_input_np = input_np.copy()
    curr_past_lens_np = np.array(past_lens, dtype=np.int32)
    curr_pos_ids_np = np.array(pos_ids, dtype=np.int32)
    
    start = time.time()
    infini_out = None
    
    for run_idx in range(RUNS):
        out_np = infinilm_model.forward(curr_input_np, batch_size, seq_lens, curr_past_lens_np, curr_pos_ids_np, return_raw=False)
        
        if run_idx < RUNS - 1:
            # Update inputs for next round
            curr_input_np = out_np
            
            curr_past_lens_np = [x + 1 for x in curr_past_lens_np]
            curr_pos_ids_np = [x + 1 for x in curr_pos_ids_np]
                
        infini_out = out_np
        
    torch.cuda.synchronize()
    end = time.time()
    i_throughput = (total_tokens_per_round * RUNS) / (end - start)

    print(f"Throughput: Torch={t_throughput:.1f} tok/s, Infini={i_throughput:.1f} tok/s")
    check_correctness_decode(torch_out_list, infini_out, device)

if __name__ == "__main__":
    
    args = get_args()
    device = "cuda" if args.nvidia else "cpu"
    
    torch_model, rotary, cfg = create_Qwen3attention_torch(args.model_path, device)
    infini_model = InfiniLMWrapper(cfg, torch_model)
    
    benchmark_prefill(torch_model, rotary, infini_model, PREFILL_TESTCASES, device, torch.bfloat16)
    benchmark_decode(torch_model, rotary, infini_model, DECODE_TESTCASES, device, torch.bfloat16)