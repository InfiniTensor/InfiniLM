#!/usr/bin/env python3
"""
M3 end-to-end comparison: InfiniLM `minimax` vs HF transformers `MiniMaxForCausalLM`.

Uses a small config with `num_local_experts = 1`, which is mathematically a dense
SwiGLU MLP in both implementations (the fused MoE runner is CUDA-only, so the
multi-expert routing path is validated on NVIDIA in CI).

Checks:
1. Full-prefill logits match at every position (HF chunked prefill vs our
   indexed recurrent op).
2. Decode-after-context logits match (HF MiniMaxCache recurrent path vs ours).
"""
import json
import sys

import torch
from transformers import MiniMaxConfig, MiniMaxForCausalLM

import infinicore
from infinicore.lib import _infinicore

sys.path.insert(0, __file__.rsplit("\\", 1)[0])
from smoke_minimax import create_engine, i2t, t2i

from infinilm.modeling_utils import _remap_minimax


def make_hf_config():
    return MiniMaxConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        hidden_act="silu",
        max_position_embeddings=64,
        rms_norm_eps=1e-5,
        num_experts_per_tok=1,
        num_local_experts=1,
        attention_dropout=0.0,
        block_size=16,
        layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 1000000.0},
    )


def hf_config_to_infinilm_dict(hf_config) -> dict:
    d = hf_config.to_dict()
    d["model_type"] = "minimax"
    d["torch_dtype"] = "float32"
    d["block"] = d.pop("block_size", 16)
    # Drop null fields (e.g. `torch_dtype: None`) that break ModelConfig::get_dtype.
    d = {k: v for k, v in d.items() if v is not None}
    return d


def load_hf_weights_into_engine(engine, dev, hf_model, hf_config):
    hf_sd = hf_model.state_dict()
    remapped = _remap_minimax(hf_sd, hf_config.to_dict())
    expected = set(engine.state_dict_keyname())
    params = {}
    keep = []
    matched = 0
    for key, tensor in remapped.items():
        if key in expected:
            t = tensor.detach().float().contiguous()
            keep.append(t)
            params[key] = t2i(t, dev)
            matched += 1
        else:
            print(f"  (skip) {key}")
    print(f"  matched {matched}/{len(remapped)} weight keys")
    engine.load_params(params, strict=False)
    engine.process_weights_after_loading()
    return matched


def run_hf_prefill(hf_model, tokens):
    ids = torch.tensor([tokens], dtype=torch.long)
    pos = torch.arange(len(tokens), dtype=torch.long).unsqueeze(0)
    mask = torch.ones(1, len(tokens), dtype=torch.long)
    out = hf_model(input_ids=ids, position_ids=pos, attention_mask=mask, use_cache=False)
    return out.logits.detach().float()


def run_hf_decode(hf_model, tokens, next_token):
    ids4 = torch.tensor([tokens], dtype=torch.long)
    pos4 = torch.arange(len(tokens), dtype=torch.long).unsqueeze(0)
    mask4 = torch.ones(1, len(tokens), dtype=torch.long)
    out4 = hf_model(input_ids=ids4, position_ids=pos4, attention_mask=mask4, use_cache=True)
    past = out4.past_key_values
    ids1 = torch.tensor([[next_token]], dtype=torch.long)
    pos1 = torch.tensor([[len(tokens)]], dtype=torch.long)
    mask5 = torch.ones(1, len(tokens) + 1, dtype=torch.long)
    out1 = hf_model(input_ids=ids1, position_ids=pos1, attention_mask=mask5, use_cache=True, past_key_values=past)
    return out1.logits.detach().float()


def run_il_prefill(engine, dev, tokens):
    n = len(tokens)
    input_ids = torch.tensor([tokens], dtype=torch.int32)
    position_ids = torch.arange(n, dtype=torch.int32).unsqueeze(0)
    past = torch.tensor([0], dtype=torch.int32)
    total = torch.tensor([n], dtype=torch.int32)
    offsets = torch.tensor([0, n], dtype=torch.int32)
    cu = torch.tensor([0, n], dtype=torch.int32)
    idx = torch.tensor([0], dtype=torch.int32)
    from smoke_minimax import make_input
    out = engine.forward(make_input(dev, input_ids, position_ids, past, total, offsets, cu, idx, idx))
    return i2t(out.logits)


def run_il_decode(engine, dev, tokens, next_token):
    from smoke_minimax import run_decode
    return run_decode(engine, dev, next_token, past_len=len(tokens), slot=0)


def main():
    hf_config = make_hf_config()
    torch.manual_seed(7)
    hf_model = MiniMaxForCausalLM(hf_config)
    hf_model.eval()

    cfg_dict = hf_config_to_infinilm_dict(hf_config)
    print("[1/4] constructing InfiniLM minimax engine (dense 1-expert path) ...")
    engine, dev = create_engine(cfg_dict)

    print("[2/4] loading HF weights via _remap_minimax ...")
    load_hf_weights_into_engine(engine, dev, hf_model, hf_config)

    tokens = [3, 7, 9, 2, 11]
    print("[3/4] comparing full-prefill logits ...")
    hf_prefill = run_hf_prefill(hf_model, tokens)
    il_prefill = run_il_prefill(engine, dev, tokens)
    print("  hf_prefill:", tuple(hf_prefill.shape), " il_prefill:", tuple(il_prefill.shape))
    err = (hf_prefill - il_prefill).abs().max().item()
    print(f"  max |hf - infinilm| (prefill, all positions) = {err:.6e}")
    assert err < 1e-2, f"prefill mismatch {err}"

    print("[4/4] comparing decode-after-context logits ...")
    # Use a fresh engine: prefill 4 tokens, then decode the 5th.
    engine2, dev2 = create_engine(cfg_dict)
    load_hf_weights_into_engine(engine2, dev2, hf_model, hf_config)
    run_il_prefill(engine2, dev2, tokens[:-1])
    hf_dec = run_hf_decode(hf_model, tokens[:-1], tokens[-1])
    il_dec = run_il_decode(engine2, dev2, tokens[:-1], tokens[-1])
    err2 = (hf_dec - il_dec).abs().max().item()
    print(f"  max |hf - infinilm| (decode) = {err2:.6e}")
    assert err2 < 1e-2, f"decode mismatch {err2}"

    print("PASS: minimax end-to-end matches HF transformers")
    return 0


if __name__ == "__main__":
    sys.exit(main())



