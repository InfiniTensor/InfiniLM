#!/usr/bin/env python3
"""Static-cache regression for `--draft-model` (engine construction must not crash).

The speculative runner reads the target engine's KV-cache block size while it is
constructed. Only the paged cache config exposes one, so combining
`--draft-model` with `cache_type="static"` used to fail engine construction with
an AttributeError, even though the same combination previously ran *without*
drafting: the static scheduler hands out no speculative cache ops, and the
runner's forward then stays on the plain target path.

This script pins the behavior on a real checkpoint:

  1. an engine built with `draft_model_path` and `cache_type="static"` starts
     (before the fix it raised while the speculative runner was constructed);
  2. it logs that requests run non-speculatively;
  3. it drafts nothing (no drafted tokens counted) and behaves exactly like the
     same engine built without `draft_model_path` — same tokens, or the same
     failure, since a static cache is a limited configuration for the models
     that carry recurrent state independently of any draft model.

Usage:
  python test/models/qwen3_5_mtp/test_speculative_static_cache.py \
      --model ~/models/Qwen3.5-0.8B --device cuda
"""

import argparse
import gc
import logging
import os
import re
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

DEFAULT_MODEL_DIR = os.path.expanduser("~/models/Qwen3.5-0.8B")
DEFAULT_DEVICE = "cuda"
DEFAULT_MAX_NEW_TOKENS = 24
DEFAULT_PROMPTS = ["The capital of France is", "1 + 1 ="]


class _Collect(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def build_engine(model_dir, draft_model_dir, device, max_new_tokens):
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
        max_cache_len=1024,
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
        description="Static KV cache + draft model regression"
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    args = parser.parse_args()

    print("=" * 70)
    print("Static Cache + Draft Model Construction Regression")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print("=" * 70)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("✗ CUDA device requested but torch.cuda is not available")
        return 1

    print("\n1. Baseline engine (no draft model, static cache)...")
    baseline = build_engine(args.model, None, args.device, args.max_new_tokens)
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
    runner_logger = logging.getLogger("infinilm.llm.model_runner.speculative_runner")
    runner_logger.addHandler(collector)
    try:
        speculative = build_engine(
            args.model, args.model, args.device, args.max_new_tokens
        )
        print("   ✓ constructed (this is the regression: it used to raise)")
        print("\n3. Behavior with and without the draft model...")
        speculative_tokens, speculative_error = run(
            speculative, DEFAULT_PROMPTS, args.max_new_tokens
        )
        runner = getattr(speculative.engine.model_runner, "speculative_runner", None)
        drafted = None if runner is None else runner.eagle_total_count
        block_size = None if runner is None else runner._cache_block_size
        speculative.close()
        del speculative
        gc.collect()
    finally:
        runner_logger.removeHandler(collector)

    warned = any("non-speculatively" in message for message in collector.messages)
    print(f"   warning about non-speculative requests: {'✓' if warned else '✗'}")
    print(f"   drafted tokens: {drafted}")
    print(f"   cache block size seen by the runner: {block_size}")

    if speculative_error is None:
        print("\n4. Comparing outputs token by token...")
        same_outcome = True
        for prompt, expected, got in zip(
            DEFAULT_PROMPTS, baseline_tokens, speculative_tokens
        ):
            match = bool(expected) and expected == got
            same_outcome = same_outcome and match
            print(f"   {'✓' if match else '✗'} {prompt!r} ({len(got)} tokens)")
            if not match:
                print(f"       baseline:    {expected}")
                print(f"       draft-model: {got}")
    else:
        print("\n4. Comparing the two runs' outcomes...")
        same_outcome = speculative_error == baseline_error
        print(f"   {'✓' if same_outcome else '✗'} draft-model run failed exactly")
        print("     like the run without a draft model:")
        print(f"     {speculative_error}")
        print("     (a static cache cannot serve this model's recurrent state,")
        print("      with or without --draft-model; that limitation is not the")
        print("      construction regression this test covers)")

    ok = same_outcome and warned and drafted == 0 and block_size is None
    print("\n" + "=" * 70)
    if ok:
        print("✓ Static-cache regression passed: construction succeeds and the")
        print("  requests behave exactly like a run without --draft-model")
    else:
        print("✗ Static-cache regression failed")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
