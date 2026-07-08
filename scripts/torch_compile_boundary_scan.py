#!/usr/bin/env python3
# Copyright (c) 2025, InfiniCore
"""PRD-04 M2: scan seq_len boundary around 8192 compile failure."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import List, Optional, Sequence

import torch

DEFAULT_SCAN_LENS = list(range(8188, 8197)) + [8448]


@dataclass
class ScanResult:
    seq_len: int
    token_match: bool
    eager_argmax: int
    compiled_argmax: int
    max_abs_diff: float
    strategy: str
    error: Optional[str] = None


def _make_input_ids(seq_len: int, vocab_size: int, device: torch.device, seed: int):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + seq_len)
    ids = torch.randint(1, vocab_size, (1, seq_len), generator=gen, dtype=torch.long)
    return ids.to(device)


def _scan_one(
    runner,
    seq_len: int,
    *,
    vocab_size: int,
    device: torch.device,
    seed: int,
    strategy: str,
) -> ScanResult:
    input_ids = _make_input_ids(seq_len, vocab_size, device, seed)
    with torch.inference_mode():
        eager_last = runner.model.forward_prefill_compile(input_ids)[:, -1, :]
        compiled_last = runner.run_prefill_last_logits(input_ids)
    diff = (eager_last.float() - compiled_last.float()).abs()
    eager_argmax = int(eager_last.argmax(dim=-1).item())
    compiled_argmax = int(compiled_last.argmax(dim=-1).item())
    return ScanResult(
        seq_len=seq_len,
        token_match=eager_argmax == compiled_argmax,
        eager_argmax=eager_argmax,
        compiled_argmax=compiled_argmax,
        max_abs_diff=float(diff.max().item()),
        strategy=strategy,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/models/9g_8b_thinking")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--scan-lens",
        default=",".join(str(x) for x in DEFAULT_SCAN_LENS),
    )
    parser.add_argument("--json-out", default="")
    parser.add_argument(
        "--try-mark-dynamic",
        action="store_true",
        help="Use torch._dynamo.mark_dynamic on seq dim",
    )
    parser.add_argument(
        "--try-eager-8192",
        action="store_true",
        help="Set INFINI_TORCH_COMPILE_8192_EAGER=1 for second pass",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    device = torch.device(args.device)
    scan_lens = [int(x.strip()) for x in args.scan_lens.split(",") if x.strip()]

    from infinilm.compile.config import TorchCompileConfig
    from infinilm.compile.env import compile_max_seq_len, torch_compile_cache_root
    from infinilm.compile.runner import TorchCompileRunner

    cfg = TorchCompileConfig(
        model_path=args.model_path,
        max_seq_len=compile_max_seq_len(),
        cache_root=torch_compile_cache_root(),
        prefix="boundary_scan",
    )

    strategies: list[tuple[str, dict]] = [
        ("eager_8192_default", {"mark_dynamic": False}),
        ("compile_8192_raw", {"mark_dynamic": False}),
    ]
    os.environ.pop("INFINI_TORCH_COMPILE_8192_EAGER", None)
    os.environ.pop("INFINI_TORCH_COMPILE_8192_PAD", None)
    if os.environ.get("INFINI_TORCH_COMPILE_8192_PAD", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        strategies.insert(0, ("pad_8448", {"mark_dynamic": False}))
    if args.try_mark_dynamic:
        strategies.append(("mark_dynamic", {"mark_dynamic": True}))
    if args.try_eager_8192:
        os.environ["INFINI_TORCH_COMPILE_8192_EAGER"] = "1"
        strategies.append(("eager_8192_explicit", {"mark_dynamic": False}))

    all_results: list[dict] = []
    first_fail: Optional[int] = None

    for strategy, kwargs in strategies:
        print(f"\n[boundary] strategy={strategy} ...", flush=True)
        try:
            t0 = time.perf_counter()
            runner = TorchCompileRunner(cfg, device=device, **kwargs)
            runner.warmup()
            vocab_size = int(runner.model.config.vocab_size)
            print(f"[boundary] warmup done in {time.perf_counter() - t0:.1f}s")
        except Exception as exc:  # noqa: BLE001
            print(f"[boundary] strategy={strategy} warmup failed: {exc}")
            all_results.append(
                {
                    "strategy": strategy,
                    "warmup_error": str(exc),
                }
            )
            continue

        for seq_len in scan_lens:
            try:
                result = _scan_one(
                    runner,
                    seq_len,
                    vocab_size=vocab_size,
                    device=device,
                    seed=args.seed,
                    strategy=strategy,
                )
            except Exception as exc:  # noqa: BLE001
                result = ScanResult(
                    seq_len=seq_len,
                    token_match=False,
                    eager_argmax=-1,
                    compiled_argmax=-1,
                    max_abs_diff=float("nan"),
                    strategy=strategy,
                    error=str(exc),
                )
            status = "PASS" if result.token_match else "FAIL"
            print(
                f"[boundary] {status} strategy={strategy} seq_len={seq_len} "
                f"argmax={result.eager_argmax}/{result.compiled_argmax} "
                f"max_abs_diff={result.max_abs_diff:.6f}"
                + (f" error={result.error}" if result.error else "")
            )
            all_results.append(asdict(result))
            if not result.token_match and first_fail is None:
                first_fail = seq_len

        del runner
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = {
        "first_failing_seq_len": first_fail,
        "scan_lens": scan_lens,
        "results": all_results,
    }
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[boundary] wrote {args.json_out}")

    return 0 if first_fail is None else 1


if __name__ == "__main__":
    sys.exit(main())
