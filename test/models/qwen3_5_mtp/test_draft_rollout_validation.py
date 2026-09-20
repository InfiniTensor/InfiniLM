#!/usr/bin/env python3
"""
Draft rollout validation for the Qwen3.5 MTP speculative branch.

Exercises the exact draft-side mechanics the speculative runner uses for
qwen3_5_mtp: a standalone draft engine built from the target checkpoint's
embedded MTP weights, driven step by step on the static attention backend
(the runner's default draft backend) with [3, num_tokens] mrope positions
and the post-norm hidden recycled between steps.

Checks:
  1. Step-0 draft forward (static backend) against the torch MTP reference.
  2. Multi-step rollout token equivalence between the static and paged
     backends (the gate-fixed static path vs the validated paged path).
  3. First-draft-token hit rate against the target's greedy continuation.
"""

import argparse
import gc
import os
import sys

try:
    import torch
    import transformers  # noqa: F401  (loaded by the sibling test helpers)
except ImportError as e:
    print(f"Error: Required packages not found. Please install: {e}")
    sys.exit(1)

try:
    import infinicore
    from infinilm.cache.cache import PagedKVCacheConfig, StaticKVCacheConfig
    from infinilm.infer_engine import InferEngine
    from infinilm.llm.model_runner.speculative_runner import (
        resolve_draft_engine_path,
    )
    from infinilm.modeling_utils import load_model_state_dict_by_file
except ImportError as e:
    print("Error: InfiniLM package not found. Please install it:")
    print(f"  Error: {e}")
    sys.exit(1)

# Reuse the reference module and input builders from the forward validation.
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _TEST_DIR)

from test_forward_validation import (  # noqa: E402
    TorchMtpReference,
    build_target_hidden,
)
from utils import infinicore_to_torch_tensor, tensor_all_close  # noqa: E402

DEFAULT_MODEL_DIR = os.path.expanduser("~/models/Qwen3.5-2B")
DEFAULT_DEVICE = "cuda"
DEFAULT_SEED = 20260915
DEFAULT_PROMPT = (
    "The capital of France is Paris. The largest planet in the solar system is"
)
DEFAULT_MAX_NEW_TOKENS = 24
DEFAULT_ROLLOUT_STEPS = 8
# bf16 both sides: tolerance covers kernel-order accumulation differences.
RTOL = 1e-2
ATOL = 1.0


def build_draft_engine(fixture, device, backend):
    """Build a draft engine on the fixture, mirroring the runner for static."""
    if backend == "static":
        cache_config = StaticKVCacheConfig(max_batch_size=1, max_cache_len=256)
        attention_backend = "default"
    else:
        cache_config = PagedKVCacheConfig(
            num_blocks=2, block_size=256, max_batch_size=1
        )
        attention_backend = "paged-attn"
    engine = InferEngine(
        model_path=fixture,
        device=infinicore.device(device, 0),
        cache_config=cache_config,
        attention_backend=attention_backend,
    )
    load_model_state_dict_by_file(engine, fixture, dtype=engine.dtype)
    return engine


def draft_step(engine, token, position, hidden, step, backend):
    """One serial draft step, mirroring _draft_eagle_tokens_batch (batch=1)."""
    kwargs = {
        "input_ids": infinicore.from_list([[token]], dtype=infinicore.int64),
        "position_ids": infinicore.from_list([[position]] * 3, dtype=infinicore.int64),
        "past_kv_lengths": infinicore.from_list([step], dtype=infinicore.int32),
        "total_kv_lengths": infinicore.from_list([step + 1], dtype=infinicore.int32),
        "input_offsets": infinicore.from_list([0, 1], dtype=infinicore.int32),
        "cu_seqlens": infinicore.from_list([0, step + 1], dtype=infinicore.int32),
        "target_hidden_states": hidden,
        "temperature": 1.0,
        "top_k": 1,
        "top_p": 1.0,
    }
    if backend == "paged":
        kwargs["block_tables"] = infinicore.from_list([[0]], dtype=infinicore.int32)
        kwargs["slot_mapping"] = infinicore.from_list([step], dtype=infinicore.int64)
    output = engine.forward_raw(**kwargs)
    # The static draft path samples one token per batch row: output_ids is
    # [num_rows] (the runner indexes token_ids[job_idx] directly).
    token_ids = output["output_ids"].to_numpy().tolist()
    first = token_ids[0]
    token = int(first[0]) if isinstance(first, (list, tuple)) else int(first)
    return (
        token,
        output["hidden_states"].narrow(1, 0, 1),
        output["logits"],
    )


def run_rollout(engine, source_token, source_position, source_hidden, steps, backend):
    """Serial draft rollout; returns (tokens, per-step logits/hidden)."""
    tokens = []
    outputs = []
    current_token = source_token
    current_hidden = source_hidden
    for step in range(steps):
        token, hidden, logits = draft_step(
            engine, current_token, source_position + step, current_hidden, step, backend
        )
        tokens.append(token)
        outputs.append((token, hidden, logits))
        current_token = token
        current_hidden = hidden
    return tokens, outputs


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3.5 MTP draft rollout validation (static backend)"
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--rollout-steps", type=int, default=DEFAULT_ROLLOUT_STEPS)
    parser.add_argument("--rtol", type=float, default=RTOL)
    parser.add_argument("--atol", type=float, default=ATOL)
    args = parser.parse_args()

    print("=" * 70)
    print("Qwen3.5 MTP Draft Rollout Validation (static backend)")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("✗ CUDA requested but torch.cuda is not available")
        return 1
    if args.device.startswith("cuda"):
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.manual_seed(DEFAULT_SEED)

    print("\n1. Building inputs with HF transformers prefill...")
    seq, target_hidden, embed_weight, config = build_target_hidden(
        args.model, args.prompt, args.max_new_tokens, args.device
    )
    prompt_len = len(seq) - args.max_new_tokens
    print(
        f"   sequence length: {len(seq)} (prompt {prompt_len}),"
        f" target hidden: {tuple(target_hidden.shape)}"
    )

    print("\n2. Resolving the draft fixture through the runner's own resolver...")
    fixture = resolve_draft_engine_path(args.model)
    print(f"   fixture: {fixture}")

    print("\n3. Building the static draft engine (runner's construction)...")
    static_engine = build_draft_engine(fixture, args.device, "static")

    print("\n4. Step-0 static draft forward vs torch MTP reference...")
    config._attn_implementation = "eager"
    reference = TorchMtpReference(config, embed_weight, args.model)
    reference.load_mtp_weights()
    reference = reference.to(args.device).eval()

    source_token = seq[prompt_len - 1]
    source_position = prompt_len - 1
    source_hidden = infinicore.from_torch(
        target_hidden[:, prompt_len - 1 : prompt_len, :].contiguous().to(args.device)
    )
    with torch.no_grad():
        ref_logits, ref_hidden = reference(
            torch.tensor([[source_token]], device=args.device),
            target_hidden[:, prompt_len - 1 : prompt_len, :],
            torch.arange(source_position, source_position + 1, device=args.device)
            .view(1, 1, 1)
            .expand(3, 1, 1),
        )

    token0, hidden0, logits0_raw = draft_step(
        static_engine, source_token, source_position, source_hidden, 0, "static"
    )
    logits0 = infinicore_to_torch_tensor(logits0_raw, torch.empty(0))
    ok = True
    is_close, stats = tensor_all_close(
        ref_logits[:, -1:, :].float().cpu(),
        logits0[:, -1:, :].float().cpu(),
        rtol=args.rtol,
        atol=args.atol,
    )
    ok &= is_close
    print(
        f"   {'✓' if is_close else '✗'} logits: max_abs_diff={stats['max_abs_diff']:.6f}"
        f" mean_abs_diff={stats['mean_abs_diff']:.6f}"
        f" ref_absmax={ref_logits.abs().max().item():.4f}"
    )
    static_hidden0 = infinicore_to_torch_tensor(hidden0, torch.empty(0))
    is_close, stats = tensor_all_close(
        ref_hidden.float().cpu(),
        static_hidden0.float().cpu(),
        rtol=args.rtol,
        atol=args.atol,
    )
    ok &= is_close
    print(
        f"   {'✓' if is_close else '✗'} hidden_states:"
        f" max_abs_diff={stats['max_abs_diff']:.6f}"
        f" mean_abs_diff={stats['mean_abs_diff']:.6f}"
        f" ref_absmax={ref_hidden.abs().max().item():.4f}"
    )
    print(
        f"   argmax: ref={int(ref_logits[0, -1].argmax())} static={token0}"
        f" target_next={seq[prompt_len]}"
    )

    del reference
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    print("\n5. Building the paged draft engine (equivalence reference)...")
    paged_engine = build_draft_engine(fixture, args.device, "paged")

    print(f"\n6. {args.rollout_steps}-step rollout: static vs paged tokens...")
    static_tokens, static_outputs = run_rollout(
        static_engine,
        source_token,
        source_position,
        source_hidden,
        args.rollout_steps,
        "static",
    )
    paged_tokens, paged_outputs = run_rollout(
        paged_engine,
        source_token,
        source_position,
        source_hidden,
        args.rollout_steps,
        "paged",
    )
    print(f"   static tokens: {static_tokens}")
    print(f"   paged tokens:  {paged_tokens}")
    print(
        f"   target continuation: {seq[prompt_len : prompt_len + args.rollout_steps]}"
    )
    tokens_match = static_tokens == paged_tokens
    ok &= tokens_match
    print(f"   {'✓' if tokens_match else '✗'} rollout tokens identical across backends")
    for step, ((_, s_hidden, s_logits), (_, p_hidden, p_logits)) in enumerate(
        zip(static_outputs, paged_outputs)
    ):
        s_t = infinicore_to_torch_tensor(s_hidden, torch.empty(0)).float().cpu()
        p_t = infinicore_to_torch_tensor(p_hidden, torch.empty(0)).float().cpu()
        s_l = infinicore_to_torch_tensor(s_logits, torch.empty(0)).float().cpu()
        p_l = infinicore_to_torch_tensor(p_logits, torch.empty(0)).float().cpu()
        hidden_diff = (s_t - p_t).abs().max().item()
        logits_diff = (s_l - p_l).abs().max().item()
        print(
            f"   step {step}: static-vs-paged max_abs_diff"
            f" hidden={hidden_diff:.6f} logits={logits_diff:.6f}"
        )

    print("\n7. First-draft-token hit rate over the target sequence...")
    hits = 0
    total = 0
    for idx in range(prompt_len - 1, len(seq) - 1):
        hidden_i = infinicore.from_torch(
            target_hidden[:, idx : idx + 1, :].contiguous().to(args.device)
        )
        predicted, _, _ = draft_step(
            static_engine, seq[idx], idx, hidden_i, 0, "static"
        )
        hits += int(predicted == seq[idx + 1])
        total += 1
    rate = hits / total
    print(f"   d0 == target next token: {hits}/{total} ({100.0 * rate:.1f}%)")
    # A loose floor, not an accuracy target: a draft path that silently degraded
    # into returning something other than the head's argmax would land at or
    # near zero even on a checkpoint whose draft is real.
    if hits == 0:
        print(
            "   ✗ the draft head never predicted the target's next token, which"
            " means the draft path is not producing the head's argmax"
        )
        ok = False
    else:
        print(f"   ✓ the draft head hit the target's next token {hits} time(s)")

    print("\n" + "=" * 70)
    if ok:
        print("✓ Draft rollout validation passed")
    else:
        print("✗ Draft rollout validation failed")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
