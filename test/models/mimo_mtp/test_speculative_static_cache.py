#!/usr/bin/env python3
"""Static-cache regression for `--draft-model` on a MiMo checkpoint.

The speculative runner reads the target engine's KV-cache block size while it is
constructed. Only the paged cache config exposes one, so combining
`--draft-model` with `cache_type="static"` must still construct and run: the
static scheduler hands out no speculative cache ops, the runner's forward stays
on the plain target path, and the requests run non-speculatively.

A MiMo target is pure attention, so the static cache serves it: both runs must
complete and the run with a draft model must produce exactly the tokens of the
same engine built without one. That is a stronger requirement than the check used
for a target that carries recurrent state: such a target cannot run on a static
cache at all, with or without a draft model, so its own check accepts "both runs
fail the same way" instead. Everything runs against a tiny synthetic checkpoint
unless --model names a released one.

Usage:
  python test/models/mimo_mtp/test_speculative_static_cache.py --device cpu
  python test/models/mimo_mtp/test_speculative_static_cache.py \
      --model ~/models/MiMo-7B --device cuda
"""

import argparse
import gc
import logging
import os
import re
import sys
import tempfile

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
DEFAULT_MAX_NEW_TOKENS = 24
DEFAULT_PROMPTS = ["The capital of France is", "1 + 1 ="]
# The synthetic checkpoint is small enough to run the same regression on CPU.
SYNTHETIC_BLOCK_SIZE = 16
SYNTHETIC_NUM_BLOCKS = 8
SYNTHETIC_MAX_CACHE_LEN = 512


class _Collect(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def build_engine(model_dir, draft_model_dir, device, max_new_tokens, synthetic):
    """A single engine on the static cache, with or without a draft model."""
    return LLM(
        model_path=model_dir,
        draft_model_path=draft_model_dir,
        device=device,
        dtype="bfloat16",
        cache_type="static",
        attn_backend="default",
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


def generate(engine, prompts, max_new_tokens):
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
        results.append(list(output.outputs[0].token_ids))
    return results


def _normalize(message):
    # Request ids differ between runs; everything else must match.
    return re.sub(r"cmpl-\w+", "cmpl-<id>", message)


def run(engine, prompts, max_new_tokens):
    """Generate, reporting a failure as its message instead of raising."""
    try:
        return generate(engine, prompts, max_new_tokens), None
    except Exception as error:  # noqa: BLE001 - reported and compared below
        return None, _normalize(f"{type(error).__name__}: {error}")


def main():
    parser = argparse.ArgumentParser(
        description="Static KV cache + MiMo draft model regression"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to a released MiMo checkpoint; without it a tiny synthetic "
        "checkpoint carrying the target and its draft head is used",
    )
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    args = parser.parse_args()

    print("=" * 70)
    print("Static Cache + MiMo Draft Model Construction Regression")
    print("=" * 70)
    print(f"Model: {args.model or 'synthetic checkpoint'}")
    print(f"Device: {args.device}")
    print("=" * 70)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("✗ CUDA device requested but torch.cuda is not available")
        return 1

    model_dir = args.model
    root = None
    if model_dir is None:
        root = tempfile.mkdtemp(prefix="infinilm_mimo_static_cache_")
        write_checkpoint(root)
        model_dir = root
        print(f"\n0. Wrote the synthetic checkpoint ({root})...")

    try:
        print("\n1. Baseline engine (no draft model, static cache)...")
        baseline = build_engine(
            model_dir, None, args.device, args.max_new_tokens, root is not None
        )
        baseline_tokens, baseline_error = run(
            baseline, DEFAULT_PROMPTS, args.max_new_tokens
        )
        baseline.close()
        del baseline
        gc.collect()
        if baseline_error is None:
            print(f"   ✓ constructed and generated ({len(baseline_tokens)} prompts)")
        else:
            print("   ✓ constructed; generation stopped with:")
            print(f"     {baseline_error}")

        print("\n2. Engine with --draft-model on the static cache...")
        collector = _Collect()
        runner_logger = logging.getLogger(
            "infinilm.llm.model_runner.speculative_runner"
        )
        runner_logger.addHandler(collector)
        try:
            speculative = build_engine(
                model_dir, model_dir, args.device, args.max_new_tokens, root is not None
            )
            print("   ✓ constructed (this is the regression: it used to raise)")
            print("\n3. Behavior with and without the draft model...")
            speculative_tokens, speculative_error = run(
                speculative, DEFAULT_PROMPTS, args.max_new_tokens
            )
            runner = getattr(
                speculative.engine.model_runner, "speculative_runner", None
            )
            drafted = None if runner is None else runner.eagle_total_count
            block_size = None if runner is None else runner._cache_block_size
            family = None if runner is None else runner.draft_spec.family
            speculative.close()
            del speculative
            gc.collect()
        finally:
            runner_logger.removeHandler(collector)

        warned = any("non-speculatively" in message for message in collector.messages)
        print(f"   draft description: {family}")
        print(f"   warning about non-speculative requests: {'✓' if warned else '✗'}")
        print(f"   drafted tokens: {drafted}")
        print(f"   cache block size seen by the runner: {block_size}")

        # A MiMo target is pure attention, so the static cache must serve it with
        # and without a draft model: the two runs have to complete and produce the
        # same tokens. A failed run is not an acceptable outcome here (the
        # recurrent-state limitation that makes a static cache unusable belongs to
        # the other family's check).
        failed = baseline_error is not None or speculative_error is not None
        if failed:
            print("\n4. The runs did not both complete...")
            if baseline_error is not None:
                print(f"   ✗ without --draft-model: {baseline_error}")
            if speculative_error is not None:
                print(f"   ✗ with --draft-model:    {speculative_error}")
            print(
                "     a pure-attention target must generate on the static cache,"
                " with and without a draft model"
            )
        else:
            print("\n4. Comparing outputs token by token...")
            for prompt, expected, got in zip(
                DEFAULT_PROMPTS, baseline_tokens, speculative_tokens
            ):
                match = bool(expected) and expected == got
                print(f"   {'✓' if match else '✗'} {prompt!r} ({len(got)} tokens)")
                if not match:
                    print(f"       baseline:    {expected}")
                    print(f"       draft-model: {got}")

        same_tokens = (
            not failed
            and all(
                bool(expected) and expected == got
                for expected, got in zip(baseline_tokens, speculative_tokens)
            )
            and len(baseline_tokens) == len(DEFAULT_PROMPTS)
        )
        ok = same_tokens and warned and drafted == 0 and block_size is None
        print("\n" + "=" * 70)
        if ok:
            print("✓ Static-cache regression passed: construction succeeds and the")
            print("  requests behave exactly like a run without --draft-model")
        else:
            print("✗ Static-cache regression failed")
        print("=" * 70)
        return 0 if ok else 1
    finally:
        if root is not None:
            remove_tree(root)


if __name__ == "__main__":
    sys.exit(main())
