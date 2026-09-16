#!/usr/bin/env python3
"""
Latency benchmark for Qwen3.5 MTP speculative decoding.

Greedy-decodes a fixed prompt set twice with the InfiniLM engine on the same
GPU: once with speculation off (no --draft-model) and once with MTP
speculation on. After one untimed warmup generation, each run is timed over
multiple rounds and reported as per-prompt latency plus aggregate
tokens/second; the speculative run also reports the draft acceptance
counters.

The number of draft tokens verified per target step is configurable via
--num-draft-tokens. Losslessness is only verified for the default K=1
(see test_speculative_lossless.py); K>1 numbers are mechanism performance
references only.
"""

import argparse
import gc
import os
import statistics
import sys
import time

try:
    import torch
except ImportError as e:
    print(f"Error: Required packages not found. Please install: {e}")
    sys.exit(1)

try:
    from infinilm.llm.llm import LLM
    from infinilm.llm.sampling_params import SamplingParams
except ImportError as e:
    print("Error: InfiniLM package not found. Please install it:")
    print(f"  Error: {e}")
    sys.exit(1)

DEFAULT_MODEL_DIR = os.path.expanduser("~/models/Qwen3.5-2B")
DEFAULT_DEVICE = "cuda"
DEFAULT_MAX_NEW_TOKENS = 48
DEFAULT_ROUNDS = 3

PROMPTS = [
    "1 + 1 =",
    "def fibonacci(n):",
    "SELECT * FROM users WHERE",
    "The meaning of life is",
    "The Eiffel Tower is located in",
    "The following is a list of prime numbers: 2, 3, 5, 7,",
]


def build_engine(model_dir, draft_model_dir, device, max_new_tokens, num_draft_tokens):
    """Build the LLM engine; hybrid qwen3.5 needs paged attn, no prefix cache."""
    # Small cache footprint so the target and draft engines fit alongside
    # each other on a single consumer GPU.
    return LLM(
        model_path=model_dir,
        draft_model_path=draft_model_dir,
        num_draft_tokens=num_draft_tokens,
        device=device,
        dtype="bfloat16",
        cache_type="paged",
        attn_backend="paged-attn",
        enable_prefix_caching=False,
        max_batch_size=1,
        num_blocks=32,
        block_size=256,
        max_cache_len=1024,
        max_tokens=max_new_tokens,
        temperature=1.0,
        top_p=1.0,
        top_k=1,
    )


def close_engine(engine):
    engine.close()
    del engine
    gc.collect()


def timed_rounds(engine, prompts, max_new_tokens, rounds):
    """Greedy-decode every prompt for `rounds` rounds; return per-round timings.

    Each entry maps a prompt index to (latency seconds, generated tokens).
    """
    # Only max_tokens and ignore_eos take effect at request level.
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=1.0,
        top_p=1.0,
        top_k=1,
        ignore_eos=True,
    )
    rounds_latencies = []
    for _ in range(rounds):
        round_latencies = []
        for prompt in prompts:
            start = time.perf_counter()
            output = engine.generate(
                prompts=[prompt], sampling_params=sampling_params, use_tqdm=False
            )[0]
            elapsed = time.perf_counter() - start
            round_latencies.append((elapsed, len(output.outputs[0].token_ids)))
        rounds_latencies.append(round_latencies)
    return rounds_latencies


def report_run(name, rounds_latencies):
    """Print per-prompt and aggregate statistics over the timed rounds."""
    per_prompt = list(zip(*rounds_latencies))
    print(f"   {name} per-prompt latency (median of {len(rounds_latencies)} rounds):")
    total_seconds = 0.0
    total_tokens = 0
    for idx, samples in enumerate(per_prompt):
        latencies = [latency for latency, _ in samples]
        tokens = samples[0][1]
        total_seconds += statistics.median(latencies)
        total_tokens += tokens
        print(
            f"     [{idx}] median {statistics.median(latencies) * 1000:8.1f} ms"
            f"  mean {statistics.mean(latencies) * 1000:8.1f} ms"
            f"  ({tokens} tokens)"
        )
    print(
        f"   {name} round total: median {total_seconds * 1000:.1f} ms,"
        f" {total_tokens} tokens,"
        f" {total_tokens / total_seconds:.1f} tokens/s"
    )
    return total_seconds, total_tokens


def report_accept_stats(engine):
    """Read the speculative runner's acceptance counters, if reachable."""
    runner = getattr(engine.engine.model_runner, "speculative_runner", None)
    if runner is None:
        print("   (no speculative runner found)")
        return
    total = runner.eagle_total_count
    accepted = runner.eagle_accept_count
    rate = accepted / total if total else 0.0
    print(
        f"   accepted {accepted}/{total} drafted tokens"
        f" ({100.0 * rate:.1f}% acceptance)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3.5 MTP speculative decoding latency benchmark (greedy)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help=f"Path to the Qwen3.5 checkpoint (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help="Device for both runs, e.g. cpu or cuda (default: %(default)s)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Greedy tokens generated per prompt (default: %(default)s)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=DEFAULT_ROUNDS,
        help="Timed rounds per configuration (default: %(default)s)",
    )
    parser.add_argument(
        "--num-draft-tokens",
        type=int,
        default=1,
        help="Draft tokens verified per target step; losslessness is only "
        "verified for the default K=1 (default: %(default)s)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Qwen3.5 MTP Speculative Decoding Latency Benchmark")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(
        f"Prompts: {len(PROMPTS)} fixed inputs, {args.max_new_tokens} new tokens"
        f" each, {args.rounds} timed rounds after one warmup"
    )
    print(f"Num draft tokens: {args.num_draft_tokens}")
    if args.num_draft_tokens > 1:
        print(
            "   NOTE: losslessness is not verified for num_draft_tokens > 1;"
            " these numbers are a mechanism performance reference only."
        )
    print("=" * 70)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("✗ CUDA device requested but torch.cuda is not available")
        return 1
    if args.rounds < 1:
        print("✗ --rounds must be >= 1")
        return 1
    if args.num_draft_tokens < 1:
        print("✗ --num-draft-tokens must be >= 1")
        return 1

    print("\n1. Baseline run (speculation off)...")
    baseline = build_engine(
        args.model, None, args.device, args.max_new_tokens, args.num_draft_tokens
    )
    try:
        # One untimed generation lets kernels reach steady state before the
        # timed rounds.
        timed_rounds(baseline, PROMPTS[:1], args.max_new_tokens, 1)
        baseline_rounds = timed_rounds(
            baseline, PROMPTS, args.max_new_tokens, args.rounds
        )
    finally:
        # Release the engine before the speculative run: the target and draft
        # engines must not share the GPU with the previous run's weights.
        close_engine(baseline)
    baseline_seconds, baseline_tokens = report_run("Baseline", baseline_rounds)

    print("\n2. Speculative run (MTP draft on the same checkpoint)...")
    speculative = build_engine(
        args.model,
        args.model,
        args.device,
        args.max_new_tokens,
        args.num_draft_tokens,
    )
    try:
        timed_rounds(speculative, PROMPTS[:1], args.max_new_tokens, 1)
        speculative_rounds = timed_rounds(
            speculative, PROMPTS, args.max_new_tokens, args.rounds
        )
        print("   acceptance stats (cumulative over warmup + timed rounds):")
        report_accept_stats(speculative)
    finally:
        close_engine(speculative)
    spec_seconds, spec_tokens = report_run("Speculative", speculative_rounds)

    print("\n" + "=" * 70)
    speedup = baseline_seconds / spec_seconds if spec_seconds else 0.0
    print(
        f"Speedup (round-total median, speculative / baseline): {speedup:.3f}x"
        f"  ({baseline_tokens} vs {spec_tokens} tokens)"
    )
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
