#!/usr/bin/env python3
"""Run reproducible LFM2-1.2B inference with InfiniLM.

The script is intended for the NVIDIA and Ascend validation machines.  It keeps
sampling greedy, records exact prompt/generated token IDs, and repeats the same
prompt set in one engine process so stale KV/ShortConv state is easy to detect.

Example:
    python test/models/lfm2/run_infinilm_lfm2_real.py \
        --model /data/models/LFM2-1.2B \
        --device cuda \
        --output artifacts/lfm2_infinilm_cuda.json

To turn a Transformers reference artifact into an exact correctness check:
    python test/models/lfm2/run_infinilm_lfm2_real.py \
        --model /data/models/LFM2-1.2B \
        --device cuda \
        --prompt "Who are you?" \
        --reference artifacts/lfm2_transformers_cuda.json \
        --output artifacts/lfm2_infinilm_cuda_checked.json
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

DEFAULT_PROMPTS = (
    "Who are you?",
    "请用一句中文介绍你自己。",
    "Explain in three short points why recurrent state can reduce decoding work.",
)


def resolve_attention_backend(cache_type: str, requested: str) -> str:
    """Keep cache layout consistent with the C++ attention implementation."""
    if cache_type not in ("static", "paged"):
        raise ValueError(f"Unsupported cache type: {cache_type}")
    if requested == "default":
        return "static-attn" if cache_type == "static" else "paged-attn"
    if requested not in ("static-attn", "paged-attn", "flash-attn"):
        raise ValueError(f"Unsupported LFM2 attention backend: {requested}")
    if cache_type == "static" and requested != "static-attn":
        raise ValueError("Static cache requires the static-attn backend")
    if cache_type == "paged" and requested == "static-attn":
        raise ValueError("Paged cache cannot use the static-attn backend")
    return requested


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Local LFM2-1.2B directory")
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "npu"),
        default="cuda",
        help="InfiniCore device: cuda=NVIDIA, npu=Ascend",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        help="Prompt to run; repeat this option for multiple prompts",
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--repeat",
        type=int,
        default=2,
        help="Run the whole prompt set this many times in one engine process",
    )
    parser.add_argument("--cache-type", choices=("paged", "static"), default="paged")
    parser.add_argument("--num-blocks", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--max-cache-len", type=int, default=4096)
    parser.add_argument(
        "--attn-backend",
        default="default",
        help="default selects static-attn/paged-attn to match --cache-type",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help="JSON created by reference_lfm2_real.py; its matching prompt is checked",
    )
    parser.add_argument(
        "--extra-reference",
        action="append",
        type=Path,
        default=[],
        help="Additional reference JSONs; every matching prompt is checked",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    try:
        args.attn_backend = resolve_attention_backend(
            args.cache_type, args.attn_backend
        )
    except ValueError as error:
        parser.error(str(error))
    return args


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _environment(device: str) -> dict[str, Any]:
    import torch

    result: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
    }
    if device == "cuda" and torch.cuda.is_available():
        result["accelerator"] = torch.cuda.get_device_name(0)
    elif device == "npu":
        npu = getattr(torch, "npu", None)
        if npu is not None and npu.is_available():
            result["accelerator"] = npu.get_device_name(0)
    return result


def _load_reference(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _compare_reference(
    reference: dict[str, Any] | None, runs: list[dict[str, Any]]
) -> dict[str, Any] | None:
    if reference is None:
        return None

    prompt = reference.get("prompt")
    candidate = next((run for run in runs if run["prompt"] == prompt), None)
    if candidate is None:
        return {
            "passed": False,
            "reason": f"reference prompt was not run: {prompt!r}",
        }

    expected_prompt_ids = reference.get("input_ids")
    expected_generated_ids = reference.get("generated_token_ids")
    prompt_match = candidate["prompt_token_ids"] == expected_prompt_ids
    generated_match = candidate["generated_token_ids"] == expected_generated_ids
    return {
        "passed": prompt_match and generated_match,
        "prompt": prompt,
        "prompt_token_ids_match": prompt_match,
        "generated_token_ids_match": generated_match,
        "expected_generated_token_ids": expected_generated_ids,
        "actual_generated_token_ids": candidate["generated_token_ids"],
    }


def main() -> None:
    args = parse_args()

    if args.device == "npu":
        # Initialize CANN before importing the native InfiniLM extension.  The
        # AscendC launch stubs linked through InfiniCore are initialized at
        # dlopen time and require torch_npu's process-wide runtime setup.
        import torch
        import torch_npu  # noqa: F401

        if not torch.npu.is_available():
            raise RuntimeError("Ascend NPU is not available")

    from infinilm import LLM, SamplingParams

    prompts = args.prompt or list(DEFAULT_PROMPTS)
    sampling = SamplingParams(
        temperature=1.0,
        top_k=1,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
    )
    model = LLM(
        model_path=args.model,
        device=args.device,
        tensor_parallel_size=1,
        cache_type=args.cache_type,
        max_batch_size=1,
        max_tokens=args.max_new_tokens,
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        max_cache_len=args.max_cache_len,
        temperature=1.0,
        top_k=1,
        top_p=1.0,
        enable_graph=False,
        attn_backend=args.attn_backend,
        enable_prefix_caching=False,
    )

    runs: list[dict[str, Any]] = []
    try:
        # The repetition loop is outside the prompt loop, giving A/B/C/A/B/C.
        # A repeated result must therefore survive both row reuse and intervening
        # requests without stale KV or ShortConv state contamination.
        for repetition in range(args.repeat):
            for prompt_index, prompt in enumerate(prompts):
                messages = [{"role": "user", "content": prompt}]
                start = time.perf_counter()
                request = model.chat(
                    messages=messages,
                    sampling_params=sampling,
                    use_tqdm=False,
                )[0]
                elapsed = time.perf_counter() - start
                completion = request.outputs[0]
                run = {
                    "repetition": repetition,
                    "prompt_index": prompt_index,
                    "prompt": prompt,
                    "prompt_token_ids": request.prompt_token_ids,
                    "generated_token_ids": completion.token_ids,
                    "generated_text": completion.text,
                    "finish_reason": _enum_value(completion.finish_reason),
                    "elapsed_seconds": elapsed,
                    "generated_tokens_per_second": (
                        len(completion.token_ids) / elapsed if elapsed > 0 else None
                    ),
                }
                runs.append(run)
                print(json.dumps(run, ensure_ascii=False))
    finally:
        model.close()

    sequences_by_prompt: dict[int, list[list[int]]] = {}
    for run in runs:
        sequences_by_prompt.setdefault(run["prompt_index"], []).append(
            run["generated_token_ids"]
        )
    repeat_consistent = all(
        all(sequence == sequences[0] for sequence in sequences[1:])
        for sequences in sequences_by_prompt.values()
    )

    reference = _load_reference(args.reference)
    reference_comparison = _compare_reference(reference, runs)
    reference_comparisons = [reference_comparison] if reference_comparison else []
    reference_comparisons.extend(
        _compare_reference(_load_reference(path), runs) for path in args.extra_reference
    )
    result = {
        "model": args.model,
        "configuration": {
            "cache_type": args.cache_type,
            "num_blocks": args.num_blocks,
            "block_size": args.block_size,
            "max_cache_len": args.max_cache_len,
            "attention_backend": args.attn_backend,
            "max_new_tokens": args.max_new_tokens,
            "repeat": args.repeat,
            "decoding": "greedy",
            "dtype": "from model config",
        },
        "environment": _environment(args.device),
        "repeat_consistent": repeat_consistent,
        "reference_comparison": reference_comparison,
        "reference_comparisons": reference_comparisons,
        "runs": runs,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if not repeat_consistent:
        raise SystemExit("Repeated greedy outputs differ; cache state may be stale")
    if any(not comparison["passed"] for comparison in reference_comparisons):
        raise SystemExit("InfiniLM token IDs differ from the Transformers reference")


if __name__ == "__main__":
    main()
