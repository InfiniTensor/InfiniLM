#!/usr/bin/env python3
"""
Draft rollout validation for the MiMo MTP speculative branch.

Exercises the exact draft-side mechanics the speculative runner uses for
mimo_mtp: a standalone draft engine built from the target checkpoint's embedded
MTP weights, driven step by step on the static attention backend (the runner's
default draft backend) with the description's position-id layout and the
post-norm hidden recycled between steps.

Everything runs against a tiny synthetic checkpoint that carries both the target
and the published draft head, so the check needs no downloaded weights; the
draft fixture is resolved by the runner's own resolver.

Checks:
  1. Step-0 draft forward (static backend) against the torch MTP reference.
  2. Multi-step rollout token equivalence between the static and paged
     backends (the runner's static path vs the validated paged path).
  3. First-draft-token hit rate against the target's greedy continuation.
"""

import argparse
import gc
import os
import sys
import tempfile

try:
    import torch
    import transformers  # noqa: F401  (loaded by the sibling test helpers)
except ImportError as e:
    print(f"Error: Required packages not found. Please install: {e}")
    sys.exit(1)

try:
    import infinicore
    from infinilm.cache.cache import PagedKVCacheConfig, StaticKVCacheConfig
    from infinilm.draft_spec import draft_position_ids, resolve_draft
    from infinilm.infer_engine import InferEngine
    from infinilm.modeling_utils import load_model_state_dict_by_file
except ImportError as e:
    print("Error: InfiniLM package not found. Please install it:")
    print(f"  Error: {e}")
    sys.exit(1)

# Reuse the reference module, the fixture builders and the tensor helpers.
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_TEST_DIR, "..", "llama"))
sys.path.insert(0, _TEST_DIR)

from test_forward_validation import (  # noqa: E402
    TorchMimoMtpReference,
    build_reference_config,
)
from test_speculative_lossless import (  # noqa: E402
    PAGED_HEAD_DIM,
    remove_tree,
    write_checkpoint,
)
from utils import infinicore_to_torch_tensor, tensor_all_close  # noqa: E402

DEFAULT_DEVICE = "cuda"
DEFAULT_ROLLOUT_STEPS = 8
# The target pass runs on the paged cache, which has no CPU backend.
TARGET_BLOCK_SIZE = 16
TARGET_NUM_BLOCKS = 8
# The draft rollout starts from a position the prompt covers, so the draft's own
# keys and values sit in a cache of their own.
DRAFT_PAGED_NUM_BLOCKS = 2
DRAFT_PAGED_BLOCK_SIZE = 256
DRAFT_STATIC_MAX_LEN = 256
# The character-level tokenizer of the synthetic checkpoint spells the prompt as
# one token per character.
PROMPT_TOKENS = list(range(ord("a"), ord("p") + 1))
START_POSITION = 3
# bf16 both sides: tolerance covers kernel-order accumulation differences. The
# absolute term is calibrated to this checkpoint's logit scale (|logits| <= ~1);
# the component-level forward check carries the tight float32 tolerance that a
# wrong published-key mapping fails by a wide margin.
RTOL = 1e-2
ATOL = 0.05


def build_target_engine(checkpoint, device):
    engine = InferEngine(
        model_path=checkpoint,
        device=infinicore.device(device, 0),
        cache_config=PagedKVCacheConfig(
            num_blocks=TARGET_NUM_BLOCKS,
            block_size=TARGET_BLOCK_SIZE,
            max_batch_size=1,
        ),
        attention_backend="paged-attn",
    )
    load_model_state_dict_by_file(engine, checkpoint, dtype=engine.dtype)
    return engine


def decode_token(output):
    """The single sampled token of a one-row batch."""
    token_ids = output["output_ids"].to_numpy().tolist()
    first = token_ids[0]
    return int(first[0]) if isinstance(first, (list, tuple)) else int(first)


def target_pass(engine, prompt_tokens, steps, device):
    """Greedy continuation of the prompt and the hidden state per position.

    The hidden state at position i is the one the draft consumes to predict the
    token at i + 1, so the two lists stay aligned with the token sequence.
    """
    block_table = list(range(TARGET_NUM_BLOCKS))
    sequence = list(prompt_tokens)
    hiddens = []

    output = engine.forward_raw(
        input_ids=infinicore.from_list([sequence], dtype=infinicore.int64),
        position_ids=infinicore.from_list(
            list(range(len(sequence))), dtype=infinicore.int64
        ),
        past_kv_lengths=infinicore.from_list([0], dtype=infinicore.int32),
        total_kv_lengths=infinicore.from_list([len(sequence)], dtype=infinicore.int32),
        input_offsets=infinicore.from_list([0, len(sequence)], dtype=infinicore.int32),
        cu_seqlens=infinicore.from_list([0, len(sequence)], dtype=infinicore.int32),
        block_tables=infinicore.from_list([block_table], dtype=infinicore.int32),
        slot_mapping=infinicore.from_list(
            list(range(len(sequence))), dtype=infinicore.int64
        ),
        temperature=1.0,
        top_k=1,
        top_p=1.0,
    )
    hidden = infinicore_to_torch_tensor(output["hidden_states"], torch.empty(0))
    hiddens.extend(hidden[:, index : index + 1, :] for index in range(len(sequence)))
    sequence.append(decode_token(output))

    for _ in range(steps):
        position = len(sequence) - 1
        output = engine.forward_raw(
            input_ids=infinicore.from_list([[sequence[-1]]], dtype=infinicore.int64),
            position_ids=infinicore.from_list([[position]], dtype=infinicore.int64),
            past_kv_lengths=infinicore.from_list([position], dtype=infinicore.int32),
            total_kv_lengths=infinicore.from_list(
                [position + 1], dtype=infinicore.int32
            ),
            input_offsets=infinicore.from_list([0, 1], dtype=infinicore.int32),
            cu_seqlens=infinicore.from_list([0, position + 1], dtype=infinicore.int32),
            block_tables=infinicore.from_list([block_table], dtype=infinicore.int32),
            slot_mapping=infinicore.from_list([position], dtype=infinicore.int64),
            temperature=1.0,
            top_k=1,
            top_p=1.0,
        )
        hiddens.append(
            infinicore_to_torch_tensor(output["hidden_states"], torch.empty(0))
        )
        sequence.append(decode_token(output))

    return sequence, hiddens


def build_draft_engine(fixture, device, backend):
    """Build a draft engine on the fixture, mirroring the runner for static."""
    if backend == "static":
        cache_config = StaticKVCacheConfig(
            max_batch_size=1, max_cache_len=DRAFT_STATIC_MAX_LEN
        )
        attention_backend = "default"
    else:
        cache_config = PagedKVCacheConfig(
            num_blocks=DRAFT_PAGED_NUM_BLOCKS,
            block_size=DRAFT_PAGED_BLOCK_SIZE,
            max_batch_size=1,
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


def draft_step(engine, spec, token, position, hidden, step, backend):
    """One serial draft step, mirroring _draft_eagle_tokens_batch (batch=1)."""
    kwargs = {
        "input_ids": infinicore.from_list([[token]], dtype=infinicore.int64),
        "position_ids": draft_position_ids(spec, [position]),
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
        output["hidden_states"].narrow(0, 0, 1),
        output["logits"],
    )


def run_rollout(
    engine, spec, source_token, source_position, source_hidden, steps, backend
):
    """Serial draft rollout; returns (tokens, per-step hidden/logits)."""
    tokens = []
    outputs = []
    current_token = source_token
    current_hidden = source_hidden
    for step in range(steps):
        token, hidden, logits = draft_step(
            engine,
            spec,
            current_token,
            source_position + step,
            current_hidden,
            step,
            backend,
        )
        tokens.append(token)
        outputs.append((token, hidden, logits))
        current_token = token
        current_hidden = hidden
    return tokens, outputs


def main():
    parser = argparse.ArgumentParser(
        description="MiMo MTP draft rollout validation (static backend)"
    )
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--rollout-steps", type=int, default=DEFAULT_ROLLOUT_STEPS)
    parser.add_argument("--rtol", type=float, default=RTOL)
    parser.add_argument("--atol", type=float, default=ATOL)
    args = parser.parse_args()

    print("=" * 70)
    print("MiMo MTP Draft Rollout Validation (static backend)")
    print("=" * 70)
    print("Checkpoint: synthetic, no downloaded weights")
    print(f"Device: {args.device}")
    print(f"Rollout steps: {args.rollout_steps}")
    print("=" * 70)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("✗ CUDA requested but torch.cuda is not available")
        print("  the paged draft backend has no CPU implementation")
        return 1

    root = tempfile.mkdtemp(prefix="infinilm_mimo_rollout_")
    fixture = None
    try:
        print("\n1. Writing the synthetic checkpoint (target + embedded draft)...")
        weights = write_checkpoint(root)
        hidden_size = weights["model.embed_tokens.weight"].shape[1]
        print(f"   checkpoint: {root}")
        print(f"   hidden size: {hidden_size}, vocab: {len(weights['lm_head.weight'])}")

        print("\n2. Resolving the draft fixture via the runner's own resolver...")
        checkpoint = resolve_draft(root)
        fixture = checkpoint.engine_path
        print(f"   family: {checkpoint.spec.family}")
        print(f"   fixture: {fixture}")

        print("\n3. Target pass: greedy continuation and hidden states...")
        target_engine = build_target_engine(root, args.device)
        sequence, hiddens = target_pass(
            target_engine, PROMPT_TOKENS, args.rollout_steps + 1, args.device
        )
        del target_engine
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"   sequence: {sequence}")
        print(f"   target continuation: {sequence[len(PROMPT_TOKENS) :]}")

        source_token = sequence[START_POSITION]
        source_hidden = hiddens[START_POSITION].to(args.device).contiguous()

        print("\n4. Step-0 static draft forward vs torch MTP reference...")
        reference = TorchMimoMtpReference(
            build_reference_config(head_dim=PAGED_HEAD_DIM),
            weights["model.embed_tokens.weight"],
            weights["lm_head.weight"],
        )
        reference.load_mtp_weights(weights)
        reference = reference.to(args.device).eval()
        with torch.no_grad():
            ref_logits, ref_hidden = reference(
                torch.tensor([[source_token]], device=args.device),
                source_hidden.float(),
                torch.tensor([[START_POSITION]], device=args.device),
            )

        static_engine = build_draft_engine(fixture, args.device, "static")
        token0, hidden0, logits0_raw = draft_step(
            static_engine,
            checkpoint.spec,
            source_token,
            START_POSITION,
            infinicore.from_torch(source_hidden),
            0,
            "static",
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
            f" target_next={sequence[START_POSITION + 1]}"
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
            checkpoint.spec,
            source_token,
            START_POSITION,
            infinicore.from_torch(source_hidden),
            args.rollout_steps,
            "static",
        )
        paged_tokens, paged_outputs = run_rollout(
            paged_engine,
            checkpoint.spec,
            source_token,
            START_POSITION,
            infinicore.from_torch(source_hidden),
            args.rollout_steps,
            "paged",
        )
        print(f"   static tokens: {static_tokens}")
        print(f"   paged tokens:  {paged_tokens}")
        tokens_match = static_tokens == paged_tokens
        ok &= tokens_match
        print(
            f"   {'✓' if tokens_match else '✗'} rollout tokens identical across backends"
        )
        for step, ((_, s_hidden, s_logits), (_, p_hidden, p_logits)) in enumerate(
            zip(static_outputs, paged_outputs)
        ):
            s_t = infinicore_to_torch_tensor(s_hidden, torch.empty(0)).float().cpu()
            p_t = infinicore_to_torch_tensor(p_hidden, torch.empty(0)).float().cpu()
            s_l = infinicore_to_torch_tensor(s_logits, torch.empty(0)).float().cpu()
            p_l = infinicore_to_torch_tensor(p_logits, torch.empty(0)).float().cpu()
            print(
                f"   step {step}: static-vs-paged max_abs_diff"
                f" hidden={(s_t - p_t).abs().max().item():.6f}"
                f" logits={(s_l - p_l).abs().max().item():.6f}"
            )

        print("\n7. First-draft-token hit rate over the target sequence...")
        hits = 0
        total = 0
        for index in range(len(PROMPT_TOKENS) - 1, len(hiddens) - 1):
            hidden_index = infinicore.from_torch(
                hiddens[index].to(args.device).contiguous()
            )
            predicted, _, _ = draft_step(
                static_engine,
                checkpoint.spec,
                sequence[index],
                index,
                hidden_index,
                0,
                "static",
            )
            hits += int(predicted == sequence[index + 1])
            total += 1
        print(
            f"   d0 == target next token: {hits}/{total} ({100.0 * hits / total:.1f}%)"
        )
        print("   (random draft weights: the rate itself is not meaningful)")

        print("\n" + "=" * 70)
        if ok:
            print("✓ Draft rollout validation passed")
        else:
            print("✗ Draft rollout validation failed")
        print("=" * 70)
        return 0 if ok else 1
    finally:
        if fixture is not None:
            remove_tree(fixture)
        remove_tree(root)


if __name__ == "__main__":
    sys.exit(main())
