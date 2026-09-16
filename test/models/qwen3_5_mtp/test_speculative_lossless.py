#!/usr/bin/env python3
"""
Losslessness test for Qwen3.5 MTP speculative decoding.

Greedy-decodes a fixed prompt set twice with the InfiniLM engine: once with
speculation off (no --draft-model) and once with MTP speculation on (the
target checkpoint passed as --draft-model). The generated token id sequences
must match token by token; the speculative run also reports the draft
acceptance counters.

The number of draft tokens verified per target step is configurable via
--num-draft-tokens. Losslessness is only verified for the default K=1; for
K>1 a partial accept would leave the target's recurrent linear-attention
state ahead of the accepted sequence, so outputs are not guaranteed to be
lossless.
"""

import argparse
import gc
import os
import sys

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

DEFAULT_PROMPTS = [
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


def build_engine_or_report(
    model_dir, draft_model_dir, device, max_new_tokens, num_draft_tokens
):
    """Build the engine, translating GPU-memory init failures into a report.

    The target and draft engines together need ~9GB VRAM on this checkpoint,
    which exceeds an 8GB GPU; surface that blocker instead of a raw traceback.
    """
    try:
        return build_engine(
            model_dir, draft_model_dir, device, max_new_tokens, num_draft_tokens
        )
    except RuntimeError as e:
        if "RankWorker failed to initialize" not in str(e):
            raise
        print("   ✗ Engine construction failed on GPU memory (RankWorker init):")
        print(
            "     the target and draft engines together need ~9GB VRAM for"
            " this checkpoint; run on a >=12GB GPU or close other GPU"
            " processes and retry."
        )
        sys.exit(1)


def generate_outputs(engine, prompts, max_new_tokens):
    """Greedy-decode every prompt and return (prompt_ids, token_ids) pairs."""
    # Only max_tokens and ignore_eos take effect at request level.
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=1.0,
        top_p=1.0,
        top_k=1,
        ignore_eos=True,
    )
    results = []
    for prompt in prompts:
        output = engine.generate(
            prompts=[prompt], sampling_params=sampling_params, use_tqdm=False
        )[0]
        results.append(
            (list(output.prompt_token_ids), list(output.outputs[0].token_ids))
        )
    return results


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
        description="Qwen3.5 MTP speculative losslessness test (greedy, token-exact)"
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
        "--num-draft-tokens",
        type=int,
        default=1,
        help="Draft tokens verified per target step; losslessness is only "
        "verified for the default K=1 (default: %(default)s)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Qwen3.5 MTP Speculative Decoding Losslessness Test")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(
        f"Prompts: {len(DEFAULT_PROMPTS)} fixed inputs, {args.max_new_tokens} new tokens each"
    )
    print(f"Num draft tokens: {args.num_draft_tokens}")
    if args.num_draft_tokens > 1:
        print(
            "   NOTE: losslessness is not verified for num_draft_tokens > 1;"
            " partial-accept consistency is tracked separately."
        )
    print("=" * 70)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("✗ CUDA device requested but torch.cuda is not available")
        return 1
    if args.max_new_tokens < 32:
        print("✗ --max-new-tokens must be >= 32 for the losslessness check")
        return 1
    if args.num_draft_tokens < 1:
        print("✗ --num-draft-tokens must be >= 1")
        return 1

    print("\n1. Baseline run (speculation off)...")
    baseline = build_engine_or_report(
        args.model, None, args.device, args.max_new_tokens, args.num_draft_tokens
    )
    try:
        baseline_results = generate_outputs(
            baseline, DEFAULT_PROMPTS, args.max_new_tokens
        )
    finally:
        # Release the engine before the speculative run: the target and draft
        # engines must not share the GPU with the previous run's weights.
        baseline.close()
        del baseline
        gc.collect()
    print(f"   ✓ {len(baseline_results)} prompts generated")

    print("\n2. Speculative run (MTP draft on the same checkpoint)...")
    speculative = build_engine_or_report(
        args.model,
        args.model,
        args.device,
        args.max_new_tokens,
        args.num_draft_tokens,
    )
    try:
        speculative_results = generate_outputs(
            speculative, DEFAULT_PROMPTS, args.max_new_tokens
        )
        print("   acceptance stats:")
        report_accept_stats(speculative)
    finally:
        speculative.close()
        del speculative
        gc.collect()
    print(f"   ✓ {len(speculative_results)} prompts generated")

    print("\n3. Comparing outputs token by token...")
    if len(baseline_results) != len(DEFAULT_PROMPTS) or len(speculative_results) != len(
        DEFAULT_PROMPTS
    ):
        print(
            "✗ Internal error: expected one result per prompt"
            f" (baseline={len(baseline_results)},"
            f" speculative={len(speculative_results)})"
        )
        return 1
    all_match = True
    for prompt, (_, baseline_ids), (_, speculative_ids) in zip(
        DEFAULT_PROMPTS, baseline_results, speculative_results
    ):
        # Two empty sequences would compare equal; treat them as a failure.
        match = (
            bool(baseline_ids)
            and bool(speculative_ids)
            and (baseline_ids == speculative_ids)
        )
        all_match = all_match and match
        status = "✓" if match else "✗"
        print(f"   {status} {prompt!r} ({len(speculative_ids)} tokens)")
        if not match:
            if len(baseline_ids) != len(speculative_ids):
                print(
                    f"       length mismatch: baseline={len(baseline_ids)},"
                    f" speculative={len(speculative_ids)}"
                )
            first_div = next(
                (
                    i
                    for i, (a, b) in enumerate(zip(baseline_ids, speculative_ids))
                    if a != b
                ),
                min(len(baseline_ids), len(speculative_ids)),
            )
            print(f"       first divergence at position {first_div}")
            print(f"       baseline:    {baseline_ids}")
            print(f"       speculative: {speculative_ids}")

    print("\n" + "=" * 70)
    if all_match:
        print("✓ Losslessness passed: speculative output matches the baseline")
    else:
        print("✗ Losslessness failed: outputs diverge")
    print("=" * 70)
    return 0 if all_match else 1


if __name__ == "__main__":
    sys.exit(main())
