#!/usr/bin/env python3
"""
Latency benchmark for MiMo MTP speculative decoding.

Greedy-decodes a fixed prompt set twice with the InfiniLM engine on the same
GPU: once with speculation off (no --draft-model) and once with MTP speculation
on. After one untimed warmup generation, each run is timed over multiple rounds
and reported as per-prompt latency plus aggregate tokens/second; the speculative
run also reports the draft acceptance counters.

The number of draft tokens verified per target step is configurable via
--num-draft-tokens; losslessness itself is checked token by token by
test_speculative_lossless.py.

Without --model the benchmark builds a tiny synthetic checkpoint that carries
the target and the released draft key layout, so the measurement is reproducible
without downloading weights; timings of the tiny target are a mechanism
reference, not a model benchmark. Point --model at a released MiMo checkpoint
for real numbers.
"""

import argparse
import gc
import os
import statistics
import sys
import tempfile
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

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_TEST_DIR, "..", "llama"))
sys.path.insert(0, _TEST_DIR)

from test_speculative_lossless import remove_tree, write_checkpoint  # noqa: E402

DEFAULT_DEVICE = "cuda"
DEFAULT_MAX_NEW_TOKENS = 48
DEFAULT_ROUNDS = 3
# The synthetic checkpoint is small, so the same paged configuration as the
# losslessness check keeps the cache footprint tiny.
SYNTHETIC_NUM_BLOCKS = 8
SYNTHETIC_BLOCK_SIZE = 16
SYNTHETIC_MAX_CACHE_LEN = 512

PROMPTS = [
    "1 + 1 =",
    "def fibonacci(n):",
    "SELECT * FROM users WHERE",
    "The meaning of life is",
    "The Eiffel Tower is located in",
    "The following is a list of prime numbers: 2, 3, 5, 7,",
]


def build_engine(
    model_dir, draft_model_dir, device, max_new_tokens, num_draft_tokens, synthetic
):
    """Build the LLM engine; MTP drafting needs the paged cache."""
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
        num_blocks=SYNTHETIC_NUM_BLOCKS if synthetic else 32,
        block_size=SYNTHETIC_BLOCK_SIZE if synthetic else 256,
        max_cache_len=SYNTHETIC_MAX_CACHE_LEN if synthetic else 1024,
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
        description="MiMo MTP speculative decoding latency benchmark (greedy)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to a released MiMo checkpoint; without it a tiny synthetic "
        "checkpoint carrying the target and its draft head is used",
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
        help="Draft tokens verified per target step (default: %(default)s)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("MiMo MTP Speculative Decoding Latency Benchmark")
    print("=" * 70)
    print(f"Model: {args.model or 'synthetic checkpoint'}")
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

    root = None
    if args.model is None:
        root = tempfile.mkdtemp(prefix="infinilm_mimo_latency_")
        write_checkpoint(root)
        print(f"\n0. Wrote the synthetic checkpoint ({root})...")
    model_dir = args.model or root
    synthetic = root is not None

    try:
        print("\n1. Baseline run (speculation off)...")
        baseline = build_engine(
            model_dir,
            None,
            args.device,
            args.max_new_tokens,
            args.num_draft_tokens,
            synthetic,
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
            model_dir,
            model_dir,
            args.device,
            args.max_new_tokens,
            args.num_draft_tokens,
            synthetic,
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
            f"Speedup (round-total median, baseline / speculative): {speedup:.3f}x"
            f"  ({baseline_tokens} vs {spec_tokens} tokens)"
        )
        print("=" * 70)
        return 0
    finally:
        if root is not None:
            remove_tree(root)


if __name__ == "__main__":
    sys.exit(main())
