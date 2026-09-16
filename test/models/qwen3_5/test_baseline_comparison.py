#!/usr/bin/env python3
"""
Baseline comparison test for the InfiniLM Qwen3.5 target model.

Greedy-decodes a fixed prompt set and compares the generated token id
sequences token by token: HF transformers Qwen3_5ForCausalLM (text part of
the VL checkpoint, via AutoModelForCausalLM) vs the infinilm.llm.llm.LLM
engine (C++ qwen3_5, static registry). Agreement is input-dependent (inputs
selected from a broader sweep where both sides agree); regression tripwire,
not a blanket equivalence proof.
"""

import argparse
import gc
import os
import sys

try:
    import torch
    import transformers
except ImportError as e:
    print(f"Error: Required packages not found. Please install: {e}")
    sys.exit(1)

try:
    import infinicore  # noqa: F401  (ensures the InfiniCore backend is importable)
    from infinilm.llm.llm import LLM
    from infinilm.llm.sampling_params import SamplingParams
except ImportError as e:
    print("Error: InfiniLM package not found. Please install it:")
    print("  pip install -e .")
    print(f"  Error: {e}")
    sys.exit(1)

# Reuse the generic tensor-comparison helpers from test/models/llama/.
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_TEST_DIR, "..", "llama"))

from utils import tensor_all_close  # noqa: E402

DEFAULT_MODEL_DIR = os.path.expanduser("~/models/Qwen3.5-2B")
DEFAULT_DEVICE = "cuda"
DEFAULT_MAX_NEW_TOKENS = 16

DEFAULT_PROMPTS = [
    "1 + 1 =",
    "def fibonacci(n):",
    "SELECT * FROM users WHERE",
    "The meaning of life is",
    "The Eiffel Tower is located in",
    "The following is a list of prime numbers: 2, 3, 5, 7,",
]


def generate_reference(model_dir: str, prompts, device: str, max_new_tokens: int):
    """Greedy-decode every prompt with HF transformers and return token ids."""
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16
    ).to(device)
    model.eval()

    results = []
    for prompt in prompts:
        prompt_ids = tokenizer.encode(prompt)
        inputs = {
            "input_ids": torch.tensor([prompt_ids], dtype=torch.long, device=device),
            "attention_mask": torch.ones(
                1, len(prompt_ids), dtype=torch.long, device=device
            ),
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "eos_token_id": None,
        }
        with torch.no_grad():
            output = model.generate(**inputs)
        results.append((prompt_ids, output[0][len(prompt_ids) :].tolist()))

    del model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return results


def generate_engine(model_dir: str, prompts, device: str, max_new_tokens: int):
    """Greedy-decode every prompt with the InfiniLM engine and return token ids."""
    # Hybrid cache requires the paged attention backend and no prefix caching.
    model = LLM(
        model_path=model_dir,
        device=device,
        dtype="bfloat16",
        cache_type="paged",
        attn_backend="paged-attn",
        enable_prefix_caching=False,
        num_blocks=128,
        block_size=256,
        max_tokens=max_new_tokens,
        # Engine reads sampling parameters from this config, not per-request.
        temperature=1.0,
        top_p=1.0,
        top_k=1,
    )
    # Only max_tokens and ignore_eos take effect at request level.
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=1.0,
        top_p=1.0,
        top_k=1,
        ignore_eos=True,
    )

    try:
        results = []
        for prompt in prompts:
            output = model.generate(
                prompts=[prompt], sampling_params=sampling_params, use_tqdm=False
            )[0]
            results.append(
                (list(output.prompt_token_ids), list(output.outputs[0].token_ids))
            )
    finally:
        model.close()
    return results


def compare_reference_prompt_ids(reference_results, engine_results):
    """The two sides must tokenize the raw prompts identically."""
    for (ref_prompt_ids, _), (engine_prompt_ids, _) in zip(
        reference_results, engine_results
    ):
        if ref_prompt_ids != engine_prompt_ids:
            return False
    return True


def compare_generated_sequences(reference_results, engine_results):
    """Token-by-token comparison; returns (all_match, details) for reporting."""
    all_match = True
    details = []
    for (_, ref_ids), (_, engine_ids) in zip(reference_results, engine_results):
        # Token ids fit float32 exactly (< 2**24), required for the stats helper.
        is_close, stats = tensor_all_close(
            torch.tensor(ref_ids, dtype=torch.float32),
            torch.tensor(engine_ids, dtype=torch.float32),
        )
        match = ref_ids == engine_ids
        all_match = all_match and match
        details.append((ref_ids, engine_ids, match, is_close, stats))
    return all_match, details


def run_baseline_comparison(model_dir, prompts, device, max_new_tokens):
    print("=" * 70)
    print("Qwen3.5 Baseline Comparison Test")
    print("=" * 70)
    print(f"Model: {model_dir}")
    print(f"Device: {device}")
    print(f"Prompts: {len(prompts)} fixed inputs, {max_new_tokens} new tokens each")
    print("=" * 70)

    print("\n1. Checking device availability...")
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("   ✗ CUDA device requested but torch.cuda is not available")
        return False
    print(f"   ✓ Device {device} is available")

    print("\n2. Running reference side (HF transformers)...")
    try:
        reference_results = generate_reference(
            model_dir, prompts, device, max_new_tokens
        )
        print(f"   ✓ Reference generation done for {len(reference_results)} prompts")
    except Exception as e:
        print(f"   ✗ Reference side failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n3. Running engine side (InfiniLM LLM)...")
    try:
        engine_results = generate_engine(model_dir, prompts, device, max_new_tokens)
        print(f"   ✓ Engine generation done for {len(engine_results)} prompts")
    except Exception as e:
        print(f"   ✗ Engine side failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n4. Comparing outputs...")
    if len(reference_results) != len(engine_results) or len(engine_results) != len(
        prompts
    ):
        print(
            "   ✗ Result count mismatch: prompts="
            f"{len(prompts)}, reference={len(reference_results)}, "
            f"engine={len(engine_results)}"
        )
        return False
    if not compare_reference_prompt_ids(reference_results, engine_results):
        print("   ✗ Prompt token ids differ between the two sides")
        return False
    print("   ✓ Prompt token ids match (same tokenizer output on both sides)")

    all_match, details = compare_generated_sequences(reference_results, engine_results)
    for prompt, (ref_ids, engine_ids, match, is_close, stats) in zip(prompts, details):
        status = "✓" if match else "✗"
        print(f"   {status} {prompt!r}")
        print(f"       reference: {ref_ids}")
        print(f"       engine:    {engine_ids}")
        if not match:
            if len(ref_ids) != len(engine_ids):
                print(
                    f"       length mismatch: reference={len(ref_ids)} tokens, "
                    f"engine={len(engine_ids)} tokens"
                )
            first_div = next(
                (i for i, (a, b) in enumerate(zip(ref_ids, engine_ids)) if a != b),
                min(len(ref_ids), len(engine_ids)),
            )
            diff_stats = (
                f", max_abs_diff={stats['max_abs_diff']}"
                if "max_abs_diff" in stats
                else ""
            )
            print(
                f"       first divergence at position {first_div} "
                f"(allclose={is_close}{diff_stats})"
            )

    if not all_match:
        print("\n   ✗ Generated sequences differ")
        return False

    print("\n" + "=" * 70)
    print("✓ Baseline comparison passed: agreement is input-dependent")
    print("  (regression tripwire, not a blanket equivalence proof)")
    print("=" * 70)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3.5 baseline comparison test (InfiniLM engine vs HF transformers)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help=f"Path to the Qwen3.5 model directory (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help="Device for both sides, e.g. cpu or cuda (default: %(default)s)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Greedy tokens generated per prompt (default: %(default)s)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.model):
        print(f"Error: Model directory not found: {args.model}")
        sys.exit(1)
    if args.max_new_tokens < 1:
        print(f"Error: --max-new-tokens must be positive, got {args.max_new_tokens}")
        sys.exit(1)

    try:
        success = run_baseline_comparison(
            args.model, DEFAULT_PROMPTS, args.device, args.max_new_tokens
        )
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        success = False
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
