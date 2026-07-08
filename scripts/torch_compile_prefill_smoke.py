#!/usr/bin/env python3
# Copyright (c) 2025, InfiniCore
"""PRD-04 M2: eager vs TorchCompileRunner parity on TorchLlamaPrefillModel prefill forward."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from typing import List, Optional, Sequence

import torch

DEFAULT_BUCKETS: tuple[int, ...] = (128, 512, 1024, 4096, 8192)
KNOWN_XFAIL_BUCKETS: frozenset[int] = frozenset()


@dataclass
class BucketResult:
    seq_len: int
    passed: bool
    max_abs_diff: float
    mean_abs_diff: float
    eager_ms: float
    compiled_ms: float
    eager_argmax: int = -1
    compiled_argmax: int = -1
    token_match: bool = False
    xfail: bool = False
    error: Optional[str] = None


def _parse_buckets(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _make_input_ids(
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + seq_len)
    ids = torch.randint(1, vocab_size, (1, seq_len), generator=gen, dtype=torch.long)
    return ids.to(device)


def _compare_logits(
    eager: torch.Tensor,
    compiled: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> tuple[bool, float, float]:
    eager_f = eager.float()
    compiled_f = compiled.float()
    diff = (eager_f - compiled_f).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    ok = torch.allclose(eager_f, compiled_f, rtol=rtol, atol=atol)
    return ok, max_abs, mean_abs


def _run_bucket(
    runner,
    seq_len: int,
    *,
    vocab_size: int,
    device: torch.device,
    seed: int,
    rtol: float,
    atol: float,
    warmup_iters: int,
    xfail_buckets: frozenset[int],
) -> BucketResult:
    input_ids = _make_input_ids(seq_len, vocab_size, device, seed)
    is_xfail = seq_len in xfail_buckets

    with torch.inference_mode():
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        eager_logits = runner.model.forward_prefill_compile(input_ids)
        if device.type == "cuda":
            torch.cuda.synchronize()
        eager_ms = (time.perf_counter() - t0) * 1000.0

        for _ in range(warmup_iters):
            _ = runner.run_prefill_last_logits(input_ids)
        if device.type == "cuda":
            torch.cuda.synchronize()

        t1 = time.perf_counter()
        compiled_last = runner.run_prefill_last_logits(input_ids)
        if device.type == "cuda":
            torch.cuda.synchronize()
        compiled_ms = (time.perf_counter() - t1) * 1000.0

    eager_last = eager_logits[:, -1, :]
    ok, max_abs, mean_abs = _compare_logits(
        eager_last, compiled_last, rtol=rtol, atol=atol
    )
    eager_argmax = int(eager_last.argmax(dim=-1).item())
    compiled_argmax = int(compiled_last.argmax(dim=-1).item())
    token_match = eager_argmax == compiled_argmax
    logits_ok = ok or (max_abs <= 0.2 and token_match)
    passed = logits_ok and token_match
    if is_xfail and not passed:
        passed = True
    return BucketResult(
        seq_len=seq_len,
        passed=passed,
        max_abs_diff=max_abs,
        mean_abs_diff=mean_abs,
        eager_ms=eager_ms,
        compiled_ms=compiled_ms,
        eager_argmax=eager_argmax,
        compiled_argmax=compiled_argmax,
        token_match=token_match,
        xfail=is_xfail and not token_match,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default="/models/9g_8b_thinking",
        help="HF model directory",
    )
    parser.add_argument(
        "--buckets",
        default=",".join(str(b) for b in DEFAULT_BUCKETS),
        help="Comma-separated seq_len buckets",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--atol", type=float, default=0.02)
    parser.add_argument(
        "--compile-mode",
        default="default",
        choices=("default", "reduce-overhead", "max-autotune"),
    )
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument(
        "--cache-root",
        default="",
        help="Override INFINI_TORCH_COMPILE_CACHE (default bench_results/torch_compile_cache)",
    )
    parser.add_argument(
        "--xfail-buckets",
        default="",
        help="Comma-separated buckets treated as known-xfail (e.g. 8192)",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional path to write structured results JSON",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    device = torch.device(args.device)
    buckets = _parse_buckets(args.buckets)
    xfail_buckets = (
        frozenset(_parse_buckets(args.xfail_buckets))
        if args.xfail_buckets
        else KNOWN_XFAIL_BUCKETS
    )

    print(f"[M2] model={args.model_path} device={device} buckets={buckets}")
    print(f"[M2] rtol={args.rtol} atol={args.atol} compile_mode={args.compile_mode}")

    from infinilm.compile.config import TorchCompileConfig
    from infinilm.compile.env import compile_max_seq_len, torch_compile_cache_root
    from infinilm.compile.runner import TorchCompileRunner

    cache_root = args.cache_root or torch_compile_cache_root()
    cfg = TorchCompileConfig(
        model_path=args.model_path,
        max_seq_len=compile_max_seq_len(),
        cache_root=cache_root,
    )

    t_load = time.perf_counter()
    runner = TorchCompileRunner(
        cfg,
        device=device,
        compile_mode=args.compile_mode,
    )
    print(f"[M2] warming up compile buckets {cfg.compile_sizes} ...", flush=True)
    runner.warmup()
    load_s = time.perf_counter() - t_load
    vocab_size = int(runner.model.config.vocab_size)
    print(f"[M2] runner ready in {load_s:.1f}s vocab_size={vocab_size} cache={cfg.cache_dir}")

    results: List[BucketResult] = []
    all_pass = True
    for seq_len in buckets:
        print(f"\n[M2] bucket seq_len={seq_len} ...", flush=True)
        try:
            result = _run_bucket(
                runner,
                seq_len,
                vocab_size=vocab_size,
                device=device,
                seed=args.seed,
                rtol=args.rtol,
                atol=args.atol,
                warmup_iters=args.warmup_iters,
                xfail_buckets=xfail_buckets,
            )
        except Exception as exc:  # noqa: BLE001 — smoke reports failure per bucket
            result = BucketResult(
                seq_len=seq_len,
                passed=False,
                max_abs_diff=float("nan"),
                mean_abs_diff=float("nan"),
                eager_ms=0.0,
                compiled_ms=0.0,
                error=str(exc),
            )
        results.append(result)
        status = "PASS" if result.passed else "FAIL"
        if not result.passed:
            all_pass = False
        xfail_suffix = " (xfail)" if result.xfail else ""
        err_suffix = f" error={result.error}" if result.error else ""
        print(
            f"[M2] {status}{xfail_suffix} seq_len={seq_len} "
            f"max_abs_diff={result.max_abs_diff:.6f} "
            f"mean_abs_diff={result.mean_abs_diff:.6f} "
            f"argmax={result.eager_argmax}/{result.compiled_argmax} "
            f"token_match={result.token_match} "
            f"eager_ms={result.eager_ms:.1f} compiled_ms={result.compiled_ms:.1f}"
            f"{err_suffix}"
        )

    summary = {
        "passed": all_pass,
        "model_path": args.model_path,
        "device": str(device),
        "rtol": args.rtol,
        "atol": args.atol,
        "compile_mode": args.compile_mode,
        "cache_dir": cfg.cache_dir,
        "load_seconds": load_s,
        "buckets": [asdict(r) for r in results],
    }
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[M2] wrote {args.json_out}")

    print(f"\n[M2] OVERALL: {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
