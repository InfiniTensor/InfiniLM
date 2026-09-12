#!/usr/bin/env python3
"""
M1 CPU smoke test for the InfiniLM `minimax` model (MiniMax-Text-01 / MiniMax-M2 family).

Checks:
1. The C++ `minimax` model type registers and constructs (config normalization runs).
2. Parameter names look sane.
3. Random-weight prefill + decode forward passes produce finite logits.
4. Decode-after-N-tokens logits match a full (N+1)-token prefill logits at the last position
   (linear-attention state carry consistency through the lightning layers).
"""
import ctypes
import json
import sys

import numpy as np
import torch

import infinicore
from infinicore.lib import _infinicore
from infinilm.lib import _infinilm

INFINI_DTYPE = {
    torch.float32: infinicore.float32,
    torch.int32: infinicore.int32,
    torch.int64: infinicore.int64,
}


def t2i(t: torch.Tensor, dev):
    # Return the raw _infinicore.Tensor (the engine bindings expect the pybind object).
    t = t.contiguous()
    return infinicore.from_blob(
        t.data_ptr(), list(t.shape), dtype=INFINI_DTYPE[t.dtype], device=dev
    )._underlying


def _np_dtype(infini_dtype):
    if infini_dtype == infinicore.float32:
        return np.float32
    if infini_dtype == infinicore.int32:
        return np.int32
    if infini_dtype == infinicore.int64:
        return np.int64
    raise ValueError(f"unsupported dtype {infini_dtype}")


def i2t(t) -> torch.Tensor:
    if not hasattr(t, "_underlying"):
        t = infinicore.Tensor(t)
    t = t.contiguous()
    shape = list(t.shape)
    np_dtype = _np_dtype(t.dtype)
    ctype = {np.float32: ctypes.c_float, np.int32: ctypes.c_int32, np.int64: ctypes.c_int64}[np_dtype]
    buf = (ctype * int(t.numel())).from_address(t.data_ptr())
    arr = np.frombuffer(buf, dtype=np_dtype).reshape(shape).copy()
    return torch.from_numpy(arr)


def make_config():
    return {
        "model_type": "minimax",
        "vocab_size": 128,
        "hidden_size": 32,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 8,
        "num_hidden_layers": 4,
        "intermediate_size": 64,
        "num_experts": 1,
        "num_experts_per_tok": 1,
        "shared_intermediate_size": 0,
        "rms_norm_eps": 1e-5,
        "max_position_embeddings": 64,
        "attn_type_list": [0, 0, 1, 0],  # 3 lightning + 1 softmax
        "block": 8,
        "rope_theta": 10000.0,
        "torch_dtype": "float32",}


def create_engine(cfg):
    dev = infinicore.device("cpu")
    engine = _infinilm.InferEngine(
        json.dumps(cfg),
        _infinilm.DistConfig(1),
        _infinicore.Device.Type.CPU,
        _infinilm.StaticKVCacheConfig(max_batch_size=1, max_cache_len=64),
        False,
        "static-attn",
        None,
        False,
        "sync",
        False,
    )
    return engine, dev


def load_random_weights(engine, dev, seed=0):
    torch.manual_seed(seed)
    state_dict = engine.state_dict()[0]
    params = {}
    keep = []
    for name, tensor in state_dict.items():
        shape = list(tensor.shape)
        w = (torch.randn(shape, dtype=torch.float32) * 0.02).contiguous()
        keep.append(w)
        params[name] = t2i(w, dev)
    engine.load_params(params, strict=True)
    engine.process_weights_after_loading()
    return list(state_dict.keys())


def make_input(dev, input_ids, position_ids, past, total, offsets, cu, init, final, sample_all=True):
    return _infinilm.InferEngine.Input(
        input_ids=t2i(input_ids, dev),
        position_ids=t2i(position_ids, dev),
        past_sequence_lengths=t2i(past, dev),
        total_sequence_lengths=t2i(total, dev),
        input_offsets=t2i(offsets, dev),
        cu_seqlens=t2i(cu, dev),
        mamba_init_state_indices=t2i(init, dev),
        mamba_final_state_indices=t2i(final, dev),
        sample_all_positions=sample_all,
    )


def run_prefill(engine, dev, tokens, slot=0):
    n = len(tokens)
    input_ids = torch.tensor([tokens], dtype=torch.int32)
    position_ids = torch.arange(n, dtype=torch.int32).unsqueeze(0)
    past = torch.tensor([0], dtype=torch.int32)
    total = torch.tensor([n], dtype=torch.int32)
    offsets = torch.tensor([0, n], dtype=torch.int32)
    cu = torch.tensor([0, n], dtype=torch.int32)
    init = torch.tensor([slot], dtype=torch.int32)
    final = torch.tensor([slot], dtype=torch.int32)
    out = engine.forward(make_input(dev, input_ids, position_ids, past, total, offsets, cu, init, final))
    return i2t(out.logits)


def run_decode(engine, dev, token, past_len, slot=0):
    input_ids = torch.tensor([[token]], dtype=torch.int32)
    position_ids = torch.tensor([[past_len]], dtype=torch.int32)
    past = torch.tensor([past_len], dtype=torch.int32)
    total = torch.tensor([past_len + 1], dtype=torch.int32)
    offsets = torch.tensor([0, 1], dtype=torch.int32)
    cu = torch.tensor([0, past_len + 1], dtype=torch.int32)
    init = torch.tensor([slot], dtype=torch.int32)
    final = torch.tensor([slot], dtype=torch.int32)
    out = engine.forward(make_input(dev, input_ids, position_ids, past, total, offsets, cu, init, final))
    return i2t(out.logits)


def main():
    cfg = make_config()
    print("[1/5] constructing minimax engine (model_type=minimax, 3 lightning + 1 softmax layers) ...")
    engine, dev = create_engine(cfg)

    keys = load_random_weights(engine, dev, seed=42)
    print(f"[2/5] loaded {len(keys)} random parameters")
    for k in keys:
        print("      ", k)

    print("[3/5] prefill forward (4 tokens) ...")
    logits_prefill4 = run_prefill(engine, dev, [3, 7, 9, 2], slot=0)
    assert np.isfinite(logits_prefill4.numpy()).all(), "prefill logits not finite"
    print("      prefill4 logits shape:", tuple(logits_prefill4.shape), "finite: True")

    print("[4/5] decode forward (1 token after 4-token context) ...")
    logits_decode = run_decode(engine, dev, 11, past_len=4, slot=0)
    assert np.isfinite(logits_decode.numpy()).all(), "decode logits not finite"
    print("      decode logits shape:", tuple(logits_decode.shape), "finite: True")

    print("[5/5] consistency: fresh 5-token prefill vs prefill4+decode ...")
    engine2, _ = create_engine(cfg)
    load_random_weights(engine2, dev, seed=42)
    logits_prefill5 = run_prefill(engine2, dev, [3, 7, 9, 2, 11], slot=0)

    ref = logits_prefill5[0, -1, :].float()
    got = logits_decode[0, 0, :].float()
    diff = (ref - got).abs().max().item()
    print(f"      max |prefill5[-1] - decode| = {diff:.6e}")
    assert diff < 1e-2, f"decode/prefill mismatch too large: {diff}"
    print("PASS: minimax M1 smoke test")
    return 0


if __name__ == "__main__":
    sys.exit(main())









