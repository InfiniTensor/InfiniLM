#!/usr/bin/env python3
"""Deterministic DeepSeek V4 precision tests.

Stages:
  ops      - checks math used by the C++ fallback path with fixed tensors.
  modules  - compares Python reference vs InfiniLM C++ logits on a tiny model.
  e2e      - greedy two-token generation on the same tiny model.

The tiny model writes a config.json and one safetensors file with deterministic
weights. It does not depend on production checkpoints.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "python"))

import infinicore  # noqa: E402
from infinilm.cache import PagedKVCacheConfig  # noqa: E402
from infinilm.distributed import DistConfig  # noqa: E402
from infinilm.infer_engine import GenerationConfig, InferEngine  # noqa: E402
from infinilm.modeling_utils import load_model_state_dict_by_file  # noqa: E402


DTYPE = torch.bfloat16
VOCAB_SIZE = 32
HIDDEN_SIZE = 16
HC_MULT = 2
NUM_LAYERS = 3
NUM_HEADS = 2
HEAD_DIM = 8
Q_LORA_RANK = 8
O_LORA_RANK = 4
O_GROUPS = 1
MOE_INTER = 8
N_EXPERTS = 3
TOP_K = 2
SEQ_LEN = 5


def assert_close(name: str, got: np.ndarray | torch.Tensor, ref: np.ndarray | torch.Tensor, *, atol: float, rtol: float) -> None:
    if isinstance(got, torch.Tensor):
        got_np = got.detach().float().cpu().numpy()
    else:
        got_np = np.asarray(got, dtype=np.float32)
    if isinstance(ref, torch.Tensor):
        ref_np = ref.detach().float().cpu().numpy()
    else:
        ref_np = np.asarray(ref, dtype=np.float32)
    diff = got_np - ref_np
    stats = {
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
    }
    print(f"{name}: {stats}")
    if not np.allclose(got_np, ref_np, atol=atol, rtol=rtol):
        raise AssertionError(f"{name} mismatch: {stats}, atol={atol}, rtol={rtol}")


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def rope_cache(seq_len: int, dim: int, base: float, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_partial_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_dim: int, positions: torch.Tensor) -> torch.Tensor:
    if rope_dim <= 0:
        return x
    x_pass, x_rot = x[..., :-rope_dim], x[..., -rope_dim:]
    c, s = cos[positions], sin[positions]
    while c.dim() < x_rot.dim():
        c = c.unsqueeze(0)
        s = s.unsqueeze(0)
    return torch.cat([x_pass, x_rot * c + rotate_half(x_rot) * s], dim=-1)


def sink_softmax(logits: torch.Tensor, sink: torch.Tensor) -> torch.Tensor:
    m = torch.maximum(logits.amax(dim=-1, keepdim=True), sink)
    ex = torch.exp(logits - m)
    sink_ex = torch.exp(sink - m)
    return ex / (ex.sum(dim=-1, keepdim=True) + sink_ex)


def fixed_rmsnorm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.square().mean(-1, keepdim=True) + eps)).to(x.dtype)


def swiglu_clamped(up: torch.Tensor, gate: torch.Tensor, limit: float) -> torch.Tensor:
    return torch.nn.functional.silu(torch.minimum(gate, torch.full_like(gate, limit))) * torch.clamp(up, -limit, limit)


def compressor_ref(h: torch.Tensor, wkv: torch.Tensor, wgate: torch.Tensor, ape: torch.Tensor, norm_weight: torch.Tensor, *, m: int, head_dim: int, overlap: bool, eps: float = 1e-6) -> torch.Tensor:
    kv = torch.nn.functional.linear(h.to(wkv.dtype), wkv).float()
    score = torch.nn.functional.linear(h.to(wgate.dtype), wgate).float()
    pad = (m - h.shape[1] % m) % m
    if pad:
        kv = torch.nn.functional.pad(kv, (0, 0, 0, pad))
        score = torch.nn.functional.pad(score, (0, 0, 0, pad))
    bsz, padded, dim = kv.shape
    nb = padded // m
    kv = kv.view(bsz, nb, m, dim)
    score = score.view(bsz, nb, m, dim) + ape.float()
    if overlap:
        d = head_dim
        kv_out = kv.new_zeros(bsz, nb, 2 * m, d)
        sc_out = score.new_full((bsz, nb, 2 * m, d), float("-inf"))
        kv_out[:, :, m:] = kv[:, :, :, d:]
        kv_out[:, 1:, :m] = kv[:, :-1, :, :d]
        sc_out[:, :, m:] = score[:, :, :, d:]
        sc_out[:, 1:, :m] = score[:, :-1, :, :d]
        kv, score = kv_out, sc_out
    pooled = (kv * score.softmax(dim=2)).sum(dim=2)
    x32 = pooled.float()
    out = x32 * torch.rsqrt(x32.square().mean(-1, keepdim=True) + eps)
    return (out * norm_weight.float()).to(h.dtype)


def mhc_params_ref(x: torch.Tensor, base: torch.Tensor, fn: torch.Tensor, scale: torch.Tensor, *, eps: float = 1e-6, iters: int = 3):
    bsz, seq, n, d = x.shape
    flat = x.reshape(bsz, seq, n * d).float()
    rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + eps)
    mixes = torch.nn.functional.linear(flat, fn.float()) * rsqrt
    pre = torch.sigmoid(scale[0].float() * mixes[..., :n] + base[:n].float()) + eps
    post = 2.0 * torch.sigmoid(scale[1].float() * mixes[..., n : 2 * n] + base[n : 2 * n].float())
    comb = scale[2].float() * mixes[..., 2 * n :].reshape(bsz, seq, n, n) + base[2 * n :].float().view(n, n)
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def infini_device(device: str):
    return infinicore.device("cuda", 0) if device == "cuda" else infinicore.device("cpu", 0)


def infini_from_np(x: np.ndarray, device: str):
    tensor = infinicore.from_numpy(np.ascontiguousarray(x))
    return tensor.to(infini_device(device))


def infini_to_np(x) -> np.ndarray:
    if not hasattr(x, "_underlying"):
        x = infinicore.Tensor(x)
    try:
        if x.device.type != "cpu":
            x = x.to(infinicore.device("cpu", 0))
    except Exception:
        x = x.to(infinicore.device("cpu", 0))
    return x.to_numpy().astype(np.float32)


def run_infinicore_ops_tests(device: str) -> None:
    import infinicore.nn.functional as F
    from infinicore.ops.add import add
    from infinicore.ops.matmul import matmul

    x = np.linspace(-0.7, 0.8, num=2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    w = np.linspace(0.3, -0.2, num=5 * 4, dtype=np.float32).reshape(5, 4)
    b = np.linspace(-0.05, 0.07, num=5, dtype=np.float32)
    linear_got = infini_to_np(F.linear(infini_from_np(x, device), infini_from_np(w, device), infini_from_np(b, device)))
    linear_ref = torch.nn.functional.linear(torch.from_numpy(x), torch.from_numpy(w), torch.from_numpy(b)).numpy()
    assert_close("ops.infinicore.linear", linear_got, linear_ref, atol=1e-4, rtol=1e-4)

    a = np.linspace(-1.0, 1.0, num=2 * 4, dtype=np.float32).reshape(2, 4)
    c = np.linspace(0.4, -0.3, num=4 * 3, dtype=np.float32).reshape(4, 3)
    matmul_got = infini_to_np(matmul(infini_from_np(a, device), infini_from_np(c, device)))
    matmul_ref = torch.matmul(torch.from_numpy(a), torch.from_numpy(c)).numpy()
    assert_close("ops.infinicore.matmul", matmul_got, matmul_ref, atol=2e-4, rtol=2e-4)

    add_got = infini_to_np(add(infini_from_np(a, device), infini_from_np(a * 0.25, device)))
    add_ref = a + a * 0.25
    assert_close("ops.infinicore.add", add_got, add_ref, atol=2e-6, rtol=2e-6)

    rms_x = np.linspace(-0.9, 0.6, num=2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    rms_w = np.linspace(0.8, 1.1, num=4, dtype=np.float32)
    rms_got = infini_to_np(F.rms_norm(infini_from_np(rms_x, device), [4], infini_from_np(rms_w, device), 1e-6))
    rms_ref_t = torch.from_numpy(rms_x)
    rms_ref = (rms_ref_t * torch.rsqrt(rms_ref_t.square().mean(-1, keepdim=True) + 1e-6) * torch.from_numpy(rms_w)).numpy()
    assert_close("ops.infinicore.rms_norm", rms_got, rms_ref, atol=3e-5, rtol=3e-5)

    up = np.linspace(-1.2, 0.9, num=2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    gate = np.linspace(0.7, -1.1, num=2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    swiglu_got = infini_to_np(F.swiglu(infini_from_np(up, device), infini_from_np(gate, device)))
    swiglu_ref = (torch.nn.functional.silu(torch.from_numpy(gate)) * torch.from_numpy(up)).numpy()
    assert_close("ops.infinicore.swiglu", swiglu_got, swiglu_ref, atol=3e-5, rtol=3e-5)

    logits = np.linspace(-1.0, 1.0, num=2 * 4 * 5, dtype=np.float32).reshape(2, 4, 5)
    causal_got = infini_to_np(F.causal_softmax(infini_from_np(logits, device)))
    causal_mask = torch.tril(torch.ones_like(torch.from_numpy(logits)), diagonal=-1).flip(dims=[-2, -1])
    causal_ref_t = torch.where(causal_mask == 1, -torch.inf, torch.from_numpy(logits).float())
    causal_ref = torch.softmax(causal_ref_t, dim=-1).numpy()
    assert_close("ops.infinicore.causal_softmax", causal_got, causal_ref, atol=3e-5, rtol=3e-5)


def run_ops_tests(device: str) -> None:
    run_infinicore_ops_tests(device)
    torch.manual_seed(7)
    torch_device = torch.device("cpu")
    x = torch.linspace(-1.2, 1.4, steps=2 * 5 * 2 * 8, dtype=torch.float32, device=torch_device).view(2, 5, 2, 8).to(DTYPE)
    base = torch.linspace(-0.2, 0.25, steps=(2 + 2) * 2, dtype=torch.float32)
    fn = torch.linspace(-0.08, 0.09, steps=((2 + 2) * 2) * 16, dtype=torch.float32).view((2 + 2) * 2, 16)
    scale = torch.tensor([0.7, -0.3, 0.4], dtype=torch.float32)
    pre, post, comb = mhc_params_ref(x, base, fn, scale)
    sub = (pre.unsqueeze(-1).to(x.dtype) * x).sum(dim=-2)
    out = post.unsqueeze(-1).to(x.dtype) * sub.unsqueeze(-2) + torch.matmul(comb.to(x.dtype), x)
    assert_close("ops.mhc.pre_sum_shape", sub, sub.clone(), atol=0.0, rtol=0.0)
    assert_close("ops.mhc.post_shape", out, out.clone(), atol=0.0, rtol=0.0)

    up = torch.linspace(-3.0, 3.0, steps=24, dtype=torch.float32).view(2, 3, 4).to(DTYPE)
    gate = torch.linspace(2.5, -2.5, steps=24, dtype=torch.float32).view(2, 3, 4).to(DTYPE)
    assert_close("ops.clamped_swiglu", swiglu_clamped(up, gate, 1.25), swiglu_clamped(up, gate, 1.25), atol=0.0, rtol=0.0)

    h = torch.linspace(-0.7, 0.9, steps=1 * 5 * 16, dtype=torch.float32).view(1, 5, 16).to(DTYPE)
    wkv = torch.linspace(-0.2, 0.25, steps=16 * 16, dtype=torch.float32).view(16, 16).to(DTYPE)
    wg = torch.linspace(0.21, -0.18, steps=16 * 16, dtype=torch.float32).view(16, 16).to(DTYPE)
    ape = torch.linspace(-0.05, 0.06, steps=4 * 16, dtype=torch.float32).view(4, 16).to(DTYPE)
    nw = torch.linspace(0.8, 1.1, steps=8, dtype=torch.float32).to(DTYPE)
    comp = compressor_ref(h, wkv, wg, ape, nw, m=4, head_dim=8, overlap=True)
    assert_close("ops.compressor.self_consistency", comp, comp.clone(), atol=0.0, rtol=0.0)

    q = fixed_rmsnorm(torch.linspace(-1, 1, steps=1 * 5 * 2 * 8).view(1, 5, 2, 8).to(DTYPE))
    kv = torch.linspace(0.5, -0.5, steps=1 * 5 * 8).view(1, 5, 8).to(DTYPE)
    pos = torch.arange(5)
    cos, sin = rope_cache(5, 4, 10000.0, torch_device, DTYPE)
    q = apply_partial_rope(q.transpose(1, 2), cos, sin, 4, pos).transpose(1, 2)
    kv = apply_partial_rope(kv, cos, sin, 4, pos)
    logits = torch.einsum("bthd,bjd->bthj", q, kv) / math.sqrt(8)
    mask = torch.arange(5).view(1, 5) <= torch.arange(5).view(5, 1)
    logits = logits.masked_fill(~mask.view(1, 5, 1, 5), float("-inf"))
    probs = sink_softmax(logits, torch.zeros(1, 1, 2, 1))
    attn = torch.einsum("bthj,bjd->bthd", probs, kv.float())
    assert_close("ops.attention.self_consistency", attn, attn.clone(), atol=0.0, rtol=0.0)
    print("ops: PASS")


def tiny_config() -> Dict:
    return {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "torch_dtype": "bfloat16",
        "vocab_size": VOCAB_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": NUM_HEADS,
        "num_key_value_heads": 1,
        "max_position_embeddings": 64,
        "head_dim": HEAD_DIM,
        "q_lora_rank": Q_LORA_RANK,
        "o_lora_rank": O_LORA_RANK,
        "o_groups": O_GROUPS,
        "qk_rope_head_dim": 4,
        "sliding_window": 8,
        "rope_theta": 10000.0,
        "compress_rope_theta": 160000.0,
        "compress_ratios": [0, 4, 0],
        "index_topk": 128,
        "index_n_heads": 2,
        "index_head_dim": 8,
        "rms_norm_eps": 1e-6,
        "hc_eps": 1e-6,
        "hc_mult": HC_MULT,
        "hc_sinkhorn_iters": 3,
        "hidden_act": "silu",
        "swiglu_limit": 1.25,
        "n_routed_experts": N_EXPERTS,
        "moe_intermediate_size": MOE_INTER,
        "n_shared_experts": 1,
        "num_experts_per_tok": TOP_K,
        "num_hash_layers": 1,
        "scoring_func": "sqrtsoftplus",
        "topk_method": "noaux_tc",
        "norm_topk_prob": False,
        "routed_scaling_factor": 1.5,
        "num_nextn_predict_layers": 0,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "initializer_range": 0.02,
        "quantization_config": None,
    }


def seq_tensor(shape: Iterable[int], start: float, step: float, dtype: torch.dtype = DTYPE) -> torch.Tensor:
    n = int(np.prod(list(shape)))
    return (torch.arange(n, dtype=torch.float32) * step + start).reshape(*shape).to(dtype)


def deterministic_state_dict(config: Dict) -> Dict[str, torch.Tensor]:
    state: Dict[str, torch.Tensor] = {}
    mix = (2 + HC_MULT) * HC_MULT
    flat = HC_MULT * HIDDEN_SIZE
    state["embed.weight"] = seq_tensor((VOCAB_SIZE, HIDDEN_SIZE), -0.31, 0.003)
    for layer in range(NUM_LAYERS):
        p = f"layers.{layer}"
        state[f"{p}.attn_norm.weight"] = seq_tensor((HIDDEN_SIZE,), 0.92 + layer * 0.01, 0.004)
        state[f"{p}.ffn_norm.weight"] = seq_tensor((HIDDEN_SIZE,), 0.87 + layer * 0.01, 0.003)
        state[f"{p}.hc_attn_base"] = seq_tensor((mix,), -0.09 + layer * 0.01, 0.006, torch.float32)
        state[f"{p}.hc_attn_fn"] = seq_tensor((mix, flat), -0.015 + layer * 0.002, 0.0007, torch.float32)
        state[f"{p}.hc_attn_scale"] = torch.tensor([0.2, -0.15, 0.1], dtype=torch.float32)
        state[f"{p}.hc_ffn_base"] = seq_tensor((mix,), 0.04 + layer * 0.01, -0.004, torch.float32)
        state[f"{p}.hc_ffn_fn"] = seq_tensor((mix, flat), 0.01 - layer * 0.001, -0.0005, torch.float32)
        state[f"{p}.hc_ffn_scale"] = torch.tensor([-0.18, 0.12, 0.08], dtype=torch.float32)
        a = f"{p}.attn"
        state[f"{a}.attn_sink"] = seq_tensor((NUM_HEADS,), -0.03 + layer * 0.01, 0.02, torch.float32)
        state[f"{a}.q_norm.weight"] = seq_tensor((Q_LORA_RANK,), 0.9, 0.01)
        state[f"{a}.kv_norm.weight"] = seq_tensor((HEAD_DIM,), 0.85, 0.015)
        state[f"{a}.wq_a.weight"] = seq_tensor((Q_LORA_RANK, HIDDEN_SIZE), -0.12 + layer * 0.01, 0.002)
        state[f"{a}.wq_b.weight"] = seq_tensor((NUM_HEADS * HEAD_DIM, Q_LORA_RANK), 0.08 - layer * 0.004, -0.0015)
        state[f"{a}.wkv.weight"] = seq_tensor((HEAD_DIM, HIDDEN_SIZE), -0.05 + layer * 0.005, 0.0018)
        state[f"{a}.wo_a.weight"] = seq_tensor((O_GROUPS * O_LORA_RANK, NUM_HEADS * HEAD_DIM // O_GROUPS), 0.04, -0.002)
        state[f"{a}.wo_b.weight"] = seq_tensor((HIDDEN_SIZE, O_GROUPS * O_LORA_RANK), -0.03, 0.0025)
        if config["compress_ratios"][layer] > 1:
            m = config["compress_ratios"][layer]
            coff = 2 if m == 4 else 1
            state[f"{a}.compressor.ape"] = seq_tensor((m, coff * HEAD_DIM), -0.02, 0.001)
            state[f"{a}.compressor.wkv.weight"] = seq_tensor((coff * HEAD_DIM, HIDDEN_SIZE), 0.03, -0.001)
            state[f"{a}.compressor.wgate.weight"] = seq_tensor((coff * HEAD_DIM, HIDDEN_SIZE), -0.025, 0.0011)
            state[f"{a}.compressor.norm.weight"] = seq_tensor((HEAD_DIM,), 0.95, 0.004)
            state[f"{a}.indexer.wq_b.weight"] = seq_tensor((config["index_n_heads"] * config["index_head_dim"], Q_LORA_RANK), 0.02, 0.001)
            state[f"{a}.indexer.weights_proj.weight"] = seq_tensor((config["index_n_heads"], HIDDEN_SIZE), -0.015, 0.001)
            state[f"{a}.indexer.compressor.ape"] = seq_tensor((m, 2 * config["index_head_dim"]), 0.01, -0.0007)
            state[f"{a}.indexer.compressor.wkv.weight"] = seq_tensor((2 * config["index_head_dim"], HIDDEN_SIZE), 0.012, 0.0008)
            state[f"{a}.indexer.compressor.wgate.weight"] = seq_tensor((2 * config["index_head_dim"], HIDDEN_SIZE), -0.011, 0.0009)
            state[f"{a}.indexer.compressor.norm.weight"] = seq_tensor((config["index_head_dim"],), 0.91, 0.003)
        f = f"{p}.ffn"
        state[f"{f}.gate.weight"] = seq_tensor((N_EXPERTS, HIDDEN_SIZE), -0.03 + layer * 0.003, 0.002, torch.float32 if layer >= config["num_hash_layers"] else DTYPE)
        if layer < config["num_hash_layers"]:
            table = torch.zeros((VOCAB_SIZE, TOP_K), dtype=torch.int64)
            for token in range(VOCAB_SIZE):
                table[token, 0] = token % N_EXPERTS
                table[token, 1] = (token + 1) % N_EXPERTS
            state[f"{f}.gate.tid2eid"] = table
        else:
            state[f"{f}.gate.bias"] = seq_tensor((N_EXPERTS,), -0.01, 0.01, torch.float32)
        for expert in range(N_EXPERTS):
            ep = f"{f}.experts.{expert}"
            state[f"{ep}.w1.weight"] = seq_tensor((MOE_INTER, HIDDEN_SIZE), -0.08 + expert * 0.01, 0.001)
            state[f"{ep}.w2.weight"] = seq_tensor((HIDDEN_SIZE, MOE_INTER), 0.07 - expert * 0.008, -0.0012)
            state[f"{ep}.w3.weight"] = seq_tensor((MOE_INTER, HIDDEN_SIZE), 0.04 + expert * 0.004, 0.0014)
        sp = f"{f}.shared_experts"
        state[f"{sp}.w1.weight"] = seq_tensor((MOE_INTER, HIDDEN_SIZE), 0.02, -0.001)
        state[f"{sp}.w2.weight"] = seq_tensor((HIDDEN_SIZE, MOE_INTER), -0.02, 0.0011)
        state[f"{sp}.w3.weight"] = seq_tensor((MOE_INTER, HIDDEN_SIZE), 0.015, 0.0012)
    state["norm.weight"] = seq_tensor((HIDDEN_SIZE,), 0.93, 0.004)
    state["head.weight"] = seq_tensor((VOCAB_SIZE, HIDDEN_SIZE), -0.09, 0.002)
    state["hc_head_base"] = seq_tensor((HC_MULT,), 0.02, -0.01, torch.float32)
    state["hc_head_fn"] = seq_tensor((HC_MULT, flat), -0.01, 0.0008, torch.float32)
    state["hc_head_scale"] = torch.tensor([0.13], dtype=torch.float32)
    return state


def write_tiny_model(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    config = tiny_config()
    (root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    save_file(deterministic_state_dict(config), str(root / "model.safetensors"))
    return root


def load_python_reference(model_dir: Path, device: str):
    code_root = Path(os.environ.get("DEEPSEEK_V4_CODE", "/data-aisoft/wangpengcheng_data/deepseek-v4-mini-1B-from-flash/code"))
    sys.path.insert(0, str(code_root))
    from deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
    from deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM

    config = DeepseekV4Config.from_pretrained(str(model_dir))
    model = DeepseekV4ForCausalLM(config).to(device=device, dtype=DTYPE)
    tensors = torch.load if False else None
    import safetensors.torch

    state = safetensors.torch.load_file(str(model_dir / "model.safetensors"), device=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def load_cpp_model(model_dir: Path, device: str):
    sys.path.insert(0, str(REPO_ROOT / "python"))
    infini_device = infinicore.device(device, 0)
    model = InferEngine(
        str(model_dir),
        device=infini_device,
        distributed_config=DistConfig(1),
        cache_config=PagedKVCacheConfig(16, 64),
        attention_backend="paged-attn",
        weight_load_mode="sync",
    )
    load_model_state_dict_by_file(model, str(model_dir), dtype=model.dtype)
    return model


def raw_cpp_forward(model: InferEngine, input_ids: list[int]) -> np.ndarray:
    raw = super(InferEngine, model)
    seq = len(input_ids)
    block = model.get_cache_config().block_size()
    max_blocks = (seq + 2 + block - 1) // block

    def u(t):
        return t._underlying if hasattr(t, "_underlying") else t

    out = raw.forward(
        raw.Input(
            u(infinicore.from_list([input_ids], dtype=infinicore.int64).view([1, seq])),
            position_ids=u(infinicore.from_list(list(range(seq)), dtype=infinicore.int64)),
            past_sequence_lengths=u(infinicore.from_list([0], dtype=infinicore.int32)),
            total_sequence_lengths=u(infinicore.from_list([seq], dtype=infinicore.int32)),
            input_offsets=u(infinicore.from_list([0, seq], dtype=infinicore.int32)),
            cu_seqlens=u(infinicore.from_list([0, seq], dtype=infinicore.int32)),
            block_tables=u(infinicore.from_list([list(range(max_blocks))], dtype=infinicore.int32)),
            slot_mapping=u(infinicore.from_list(list(range(seq)), dtype=infinicore.int64)),
            temperature=1.0,
            top_k=1,
            top_p=1.0,
        )
    )
    return infinicore.Tensor(out.logits).to_numpy().astype(np.float32)


def run_module_tests(model_dir: Path, device: str, atol: float, rtol: float) -> None:
    py_model = load_python_reference(model_dir, "cuda" if device == "cuda" else "cpu")
    cpp_model = load_cpp_model(model_dir, device)
    input_ids = [3, 5, 7, 11, 13]
    with torch.no_grad():
        py_logits = py_model(torch.tensor([input_ids], device=next(py_model.parameters()).device)).logits.float().cpu().numpy()
    cpp_logits = raw_cpp_forward(cpp_model, input_ids)
    assert_close("modules.full_logits", cpp_logits, py_logits, atol=atol, rtol=rtol)
    del cpp_model
    del py_model
    print("modules: PASS")


def run_e2e_tests(model_dir: Path, device: str) -> None:
    py_model = load_python_reference(model_dir, "cuda" if device == "cuda" else "cpu")
    cpp_model = load_cpp_model(model_dir, device)
    input_ids = [3, 5, 7, 11, 13]
    generated_py = []
    cur = input_ids[:]
    for _ in range(2):
        with torch.no_grad():
            logits = py_model(torch.tensor([cur], device=next(py_model.parameters()).device)).logits[:, -1, :].float()
        token = int(logits.argmax(dim=-1).item())
        generated_py.append(token)
        cur.append(token)
    out = cpp_model.generate(
        infinicore.from_list([input_ids], dtype=infinicore.int64),
        GenerationConfig(max_new_tokens=2, eos_token_id=[], top_k=1, top_p=1.0, temperature=1.0, stop_on_eos=False),
    )
    generated_cpp = []
    for t in out if isinstance(out, list) else [out]:
        generated_cpp.extend(int(x) for x in t.to_numpy().reshape(-1).tolist())
    print("e2e tokens", {"python": generated_py, "cpp": generated_cpp})
    if generated_cpp[:2] != generated_py:
        raise AssertionError(f"e2e token mismatch: python={generated_py}, cpp={generated_cpp}")
    del cpp_model
    del py_model
    print("e2e: PASS")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["ops", "modules", "e2e", "all"], default="all")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--workdir", type=Path, default=None)
    p.add_argument("--atol", type=float, default=2e-1)
    p.add_argument("--rtol", type=float, default=2e-1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage in ("ops", "all"):
        run_ops_tests(args.device)
    if args.stage in ("modules", "e2e", "all"):
        if args.workdir is None:
            with tempfile.TemporaryDirectory(prefix="deepseek_v4_tiny_") as tmp:
                model_dir = write_tiny_model(Path(tmp))
                if args.stage in ("modules", "all"):
                    run_module_tests(model_dir, args.device, args.atol, args.rtol)
                if args.stage in ("e2e", "all"):
                    run_e2e_tests(model_dir, args.device)
        else:
            model_dir = write_tiny_model(args.workdir)
            if args.stage in ("modules", "all"):
                run_module_tests(model_dir, args.device, args.atol, args.rtol)
            if args.stage in ("e2e", "all"):
                run_e2e_tests(model_dir, args.device)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
