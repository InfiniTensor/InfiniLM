#!/usr/bin/env python3
"""Validate recurrent cache reuse and record synchronized model-forward timing.

F32/static is a numerical correctness gate. BF16 full/cache comparisons are
diagnostics because GEMM and attention shapes change rounding. All modes gate
finite logits and repeatable generation. Timings exclude tokenization/loading,
include engine dispatch and device synchronization, and are not optimization data.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--cache-type", choices=("static", "paged"), default="paged")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--compare-full-steps", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1 or args.compare_full_steps < 0:
        parser.error(
            "steps must be positive; compare-full-steps must be nonnegative (0 = timing only)"
        )
    if args.dtype == "float32" and args.cache_type == "paged":
        parser.error("NVIDIA Paged F32 prefill is unsupported")

    import infinicore
    import numpy as np
    import torch
    from infinicore.lib import _infinicore as core
    from infinilm.infer_engine import InferEngine
    from infinilm.lib import _infinilm
    from infinilm.modeling_utils import load_model_state_dict_by_file
    from native_lfm2_tiny_smoke import tensor_to_numpy

    references = [
        json.loads((args.reference_dir / name).read_text(encoding="utf-8"))
        for name in (
            "lfm2_transformers_cuda.json",
            "lfm2_transformers_cuda_zh.json",
            "lfm2_transformers_cuda_long.json",
        )
    ]
    long_tokens = (
        references[0]["input_ids"] * 16
    )  # 208 tokens, spanning four 64-token pages.
    cases = [
        ("A", references[0]["input_ids"]),
        ("B", references[1]["input_ids"]),
        ("C", references[2]["input_ids"]),
        ("A", references[0]["input_ids"]),
        ("long", long_tokens),
        ("long", long_tokens),
        ("long", long_tokens),
    ]
    capacity = max(len(tokens) for _, tokens in cases) + args.steps + 32
    block_size = 64
    num_blocks = max(8, (capacity + block_size - 1) // block_size)
    cache = (
        _infinilm.StaticKVCacheConfig(1, capacity)
        if args.cache_type == "static"
        else _infinilm.PagedKVCacheConfig(num_blocks, block_size, 1)
    )
    backend = "static-attn" if args.cache_type == "static" else "paged-attn"

    def build_input(engine, tokens, past):
        total = past + len(tokens)
        positions = list(range(past, total))
        kwargs = {
            "mamba_init_state_indices": core.from_list(
                [0 if past == 0 else 1], core.DataType.I32
            ),
            "mamba_final_state_indices": core.from_list([1], core.DataType.I32),
        }
        if args.cache_type == "paged":
            kwargs.update(
                block_tables=core.from_list(
                    [list(range(num_blocks))], core.DataType.I32
                ),
                slot_mapping=core.from_list(positions, core.DataType.I64),
            )
        return engine._build_input(
            core.from_list([tokens], core.DataType.I64),
            position_ids=core.from_list([positions], core.DataType.I64),
            past_kv_lengths=core.from_list([past], core.DataType.I32),
            total_kv_lengths=core.from_list([total], core.DataType.I32),
            input_offsets=core.from_list([0, len(tokens)], core.DataType.I32),
            cu_seqlens=core.from_list([0, total], core.DataType.I32),
            sample_all_positions=False,
            **kwargs,
        )

    def float_logits(tensor):
        if args.dtype == "float32":
            array = tensor_to_numpy(core, tensor, np.float32)
        else:
            bits = tensor_to_numpy(core, tensor, np.uint16)
            array = torch.from_numpy(bits).view(torch.bfloat16).float().numpy()
        return array.reshape(-1, array.shape[-1])[-1]

    def gpu_memory():
        return int(
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            .strip()
            .splitlines()[0]
        )

    runs, previous_sequences = [], {}
    passed = True
    with tempfile.TemporaryDirectory(prefix="lfm2-cache-validation-") as temporary:
        config = json.loads((args.model / "config.json").read_text(encoding="utf-8"))
        config["torch_dtype"] = config["dtype"] = args.dtype
        (Path(temporary) / "config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        engines = []
        try:
            for _ in range(2 if args.compare_full_steps else 1):
                engine = InferEngine(
                    temporary,
                    device=infinicore.device("cuda"),
                    cache_config=cache,
                    attention_backend=backend,
                    weight_load_mode="sync",
                )
                load_model_state_dict_by_file(
                    engine, str(args.model), dtype=engine.dtype
                )
                engines.append(engine)
            cached_engine = engines[0]
            full_engine = engines[1] if args.compare_full_steps else None
            memory_after_load = gpu_memory()
            for name, prompt_ids in cases:
                history, generated, times, metrics = list(prompt_ids), [], [], []
                for step in range(args.steps):
                    chunk, past = (
                        (history, 0) if step == 0 else ([history[-1]], len(history) - 1)
                    )
                    native_input = build_input(cached_engine, chunk, past)
                    start = time.perf_counter()
                    output = _infinilm.InferEngine.forward(cached_engine, native_input)
                    core.sync_device()
                    times.append(time.perf_counter() - start)
                    actual = float_logits(output.logits)
                    if not np.isfinite(actual).all():
                        raise AssertionError("Non-finite cached logits")
                    if step < args.compare_full_steps:
                        recomputed = _infinilm.InferEngine.forward(
                            full_engine, build_input(full_engine, history, 0)
                        )
                        expected = float_logits(recomputed.logits)
                        if not np.isfinite(expected).all():
                            raise AssertionError("Non-finite recomputed logits")
                        delta = actual - expected
                        close = bool(
                            np.allclose(actual, expected, atol=1e-4, rtol=1e-4)
                        )
                        match = int(np.argmax(actual)) == int(np.argmax(expected))
                        # Stable sorting preserves the minimum-ID tie convention.
                        candidates = sorted(
                            set(
                                np.argsort(-actual, kind="stable")[:5].tolist()
                                + np.argsort(-expected, kind="stable")[:5].tolist()
                            )
                        )
                        metrics.append(
                            {
                                "step": step,
                                "history_token_ids": list(history),
                                "cached_argmax": int(np.argmax(actual)),
                                "full_argmax": int(np.argmax(expected)),
                                "candidate_scores": [
                                    {
                                        "token_id": int(candidate),
                                        "cached": float(actual[candidate]),
                                        "full": float(expected[candidate]),
                                    }
                                    for candidate in candidates
                                ],
                                "max_abs_error": float(np.abs(delta).max()),
                                "relative_l2_error": float(
                                    np.linalg.norm(delta)
                                    / max(np.linalg.norm(expected), 1e-12)
                                ),
                                "argmax_match": match,
                                "f32_tolerance_passed": close,
                            }
                        )
                        if args.dtype == "float32" and not (close and match):
                            passed = False
                    token = int(np.argmax(actual))
                    history.append(token)
                    generated.append(token)
                consistent = (
                    name not in previous_sequences
                    or generated == previous_sequences[name]
                )
                passed = passed and consistent
                previous_sequences[name] = generated
                entry = {
                    "case": name,
                    "prompt_tokens": len(prompt_ids),
                    "generated_token_ids": generated,
                    "repeat_consistent": consistent,
                    "prefill_forward_ms": times[0] * 1000,
                    "decode_forward_mean_ms": float(np.mean(times[1:])) * 1000
                    if len(times) > 1
                    else None,
                    "decode_forward_tokens_per_second": (len(times) - 1)
                    / sum(times[1:])
                    if len(times) > 1
                    else None,
                    "gpu_memory_used_mib": gpu_memory(),
                    "full_cache_comparisons": metrics,
                }
                runs.append(entry)
                print(json.dumps(entry), flush=True)
        finally:
            core.sync_device()
            engines.clear()
    result = {
        "passed": passed,
        "dtype": args.dtype,
        "cache_type": args.cache_type,
        "steps": args.steps,
        "compare_full_steps": args.compare_full_steps,
        "long_input_purpose": "synthetic repeated token history; cache stress, not text quality",
        "timing_scope": "engine forward + device synchronization; excludes metadata, tokenization, transfer and loading",
        "eos_policy": "ignore EOS for fixed-length kernel/cache stress",
        "full_cache_numeric_gate": "atol=rtol=1e-4 plus argmax equality"
        if args.dtype == "float32"
        else "diagnostic only",
        "model_engine_count": len(engines)
        if engines
        else (2 if args.compare_full_steps else 1),
        "gpu_memory_scope": "nvidia-smi device-wide used MiB; not allocator peak",
        "gpu_memory_after_load_mib": memory_after_load,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not passed:
        raise AssertionError("Cache consistency validation failed; see JSON")


if __name__ == "__main__":
    main()
