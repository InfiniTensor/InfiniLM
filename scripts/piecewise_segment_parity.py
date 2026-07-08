#!/usr/bin/env python3
# Copyright (c) 2025, InfiniCore
"""M4 Phase 1: AOT piecewise segment vs eager infiniop-mirror parity."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Optional, Sequence, Tuple

import torch

PARITY_MAX_ABS_DIFF = 0.2


@dataclass
class SegmentParityResult:
    segment: str
    layer_idx: int
    bucket: int
    valid_seq_len: int
    passed: bool
    token_match: bool
    max_abs_diff_q: float
    max_abs_diff_k: float
    max_abs_diff_v: float
    max_abs_diff: float
    mean_abs_diff: float
    eager_ms: float
    aot_ms: float
    package_path: str = ""
    error: Optional[str] = None


def _clone_inputs(inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
    return tuple(t.clone() for t in inputs)


def _run_eager(
    segment_module: torch.nn.Module,
    inputs: Tuple[torch.Tensor, ...],
    device: torch.device,
) -> Tuple[Tuple[torch.Tensor, ...], float]:
    hidden, residual, pos = inputs
    with torch.inference_mode():
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = segment_module(hidden, residual, pos)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0
    return out, ms


def _run_compiled(
    compiled_module: torch.nn.Module,
    inputs: Tuple[torch.Tensor, ...],
    device: torch.device,
) -> Tuple[Tuple[torch.Tensor, ...], float]:
    hidden, residual, pos = inputs
    with torch.inference_mode():
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = compiled_module(hidden, residual, pos)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0
    if not isinstance(out, (tuple, list)):
        out = (out,)
    return tuple(out), ms


def _run_aot(
    package_path: str,
    inputs: Tuple[torch.Tensor, ...],
    device: torch.device,
) -> Tuple[Tuple[torch.Tensor, ...], float]:
    from torch._inductor import aoti_load_package

    device_index = device.index if device.index is not None else 0
    runner = aoti_load_package(package_path, device_index=device_index)
    hidden, residual, pos = inputs
    with torch.inference_mode():
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = runner(hidden, residual, pos)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0
    if not isinstance(out, (tuple, list)):
        out = (out,)
    return tuple(out), ms


def _run_inductor(
    *,
    backend: str,
    segment_module: torch.nn.Module,
    inputs: Tuple[torch.Tensor, ...],
    device: torch.device,
    package_path: str,
    compiled_fn: Optional[torch.nn.Module] = None,
) -> Tuple[Tuple[torch.Tensor, ...], float, str]:
    if compiled_fn is not None:
        out, ms = _run_compiled(compiled_fn, inputs, device)
        return out, ms, "torch_compile"
    if backend == "torch_compile":
        from infinilm.compile.piecewise_segments import torch_compile_piecewise_segment

        compiled = torch_compile_piecewise_segment(segment_module, inputs)
        out, ms = _run_compiled(compiled, inputs, device)
        return out, ms, "torch_compile"
    if not os.path.isfile(package_path):
        raise FileNotFoundError(
            f"AOT package missing at {package_path}; run with --compile or --backend torch_compile"
        )
    out, ms = _run_aot(package_path, inputs, device)
    return out, ms, "aot_inductor"


def _compare_staging(
    eager_out: Tuple[torch.Tensor, ...],
    aot_out: Tuple[torch.Tensor, ...],
    *,
    valid_len: int,
) -> SegmentParityResult:
    from infinilm.compile.piecewise_segments import segment_output_fingerprint

    _hidden_e, _res_e, q_e, k_e, v_e = eager_out
    _hidden_a, _res_a, q_a, k_a, v_a = aot_out
    valid_len = int(valid_len)

    diff_q = (q_e.float() - q_a.float()).abs()
    diff_k = (k_e.float() - k_a.float()).abs()
    diff_v = (v_e.float() - v_a.float()).abs()
    diff_all = torch.cat(
        [
            diff_q[:, :valid_len].reshape(-1),
            diff_k[:, :valid_len].reshape(-1),
            diff_v[:, :valid_len].reshape(-1),
        ],
        dim=0,
    )
    max_q = float(diff_q[:, :valid_len].max().item())
    max_k = float(diff_k[:, :valid_len].max().item())
    max_v = float(diff_v[:, :valid_len].max().item())
    max_abs = float(diff_all.max().item())
    mean_abs = float(diff_all.mean().item())

    fp_e = segment_output_fingerprint(q_e, k_e, v_e, valid_len=valid_len)
    fp_a = segment_output_fingerprint(q_a, k_a, v_a, valid_len=valid_len)
    token_match = int(fp_e.argmax().item()) == int(fp_a.argmax().item())
    passed = max_abs <= PARITY_MAX_ABS_DIFF and token_match

    return SegmentParityResult(
        segment="pre_attn",
        layer_idx=-1,
        bucket=int(q_e.shape[1]),
        valid_seq_len=valid_len,
        passed=passed,
        token_match=token_match,
        max_abs_diff_q=max_q,
        max_abs_diff_k=max_k,
        max_abs_diff_v=max_v,
        max_abs_diff=max_abs,
        mean_abs_diff=mean_abs,
        eager_ms=0.0,
        aot_ms=0.0,
    )


def run_segment_parity(
    *,
    model_path: str,
    segment: str,
    layer_idx: int,
    bucket: int,
    device: torch.device,
    cache_root: str,
    valid_seq_len: Optional[int],
    seed: int,
    package_path: str = "",
    inductor_backend: str = "auto",
    compiled_fn: Optional[torch.nn.Module] = None,
) -> SegmentParityResult:
    from infinilm.compile.piecewise_segments import (
        build_piecewise_segment,
        load_torch_model_with_cpp_weights,
        make_segment_example_inputs,
        piecewise_inductor_package_path,
    )

    valid = int(valid_seq_len) if valid_seq_len is not None else int(bucket)
    torch_model = load_torch_model_with_cpp_weights(model_path, device)
    config = torch_model.config
    hidden_size = int(config.hidden_size)
    n_heads = int(config.num_attention_heads)
    n_kv = int(config.num_key_value_heads)
    head_dim = hidden_size // n_heads
    dtype = next(torch_model.inner.parameters()).dtype

    segment_module = build_piecewise_segment(
        torch_model,
        segment=segment,
        layer_idx=layer_idx,
        bucket=bucket,
        valid_seq_len=valid,
    ).eval()

    gen = torch.Generator(device=device)
    gen.manual_seed(seed + layer_idx * 1000 + bucket)
    hidden, residual, pos = make_segment_example_inputs(
        bucket=bucket,
        hidden_size=hidden_size,
        n_heads=n_heads,
        n_kv=n_kv,
        head_dim=head_dim,
        device=device,
        dtype=dtype,
        valid_seq_len=valid,
    )
    hidden.uniform_(-0.05, 0.05, generator=gen)
    residual.uniform_(-0.05, 0.05, generator=gen)

    eager_inputs = (hidden.clone(), residual.clone(), pos.clone())
    aot_inputs = _clone_inputs(eager_inputs)

    try:
        eager_out, eager_ms = _run_eager(segment_module, eager_inputs, device)
        pkg = package_path or piecewise_inductor_package_path(
            cache_root=cache_root,
            model_path=model_path,
            segment=segment,
            layer_idx=layer_idx,
            bucket=bucket,
        )
        backend = inductor_backend
        if backend == "auto":
            backend = "aot_inductor" if os.path.isfile(pkg) else "torch_compile"
        aot_out, aot_ms, backend_used = _run_inductor(
            backend=backend,
            segment_module=segment_module,
            inputs=aot_inputs,
            device=device,
            package_path=pkg,
            compiled_fn=compiled_fn,
        )
        result = _compare_staging(eager_out, aot_out, valid_len=valid)
        result.layer_idx = int(layer_idx)
        result.eager_ms = eager_ms
        result.aot_ms = aot_ms
        result.package_path = pkg if backend_used == "aot_inductor" else f"torch_compile:{backend_used}"
        return result
    except Exception as exc:  # noqa: BLE001
        return SegmentParityResult(
            segment=segment,
            layer_idx=int(layer_idx),
            bucket=int(bucket),
            valid_seq_len=valid,
            passed=False,
            token_match=False,
            max_abs_diff_q=float("nan"),
            max_abs_diff_k=float("nan"),
            max_abs_diff_v=float("nan"),
            max_abs_diff=float("nan"),
            mean_abs_diff=float("nan"),
            eager_ms=0.0,
            aot_ms=0.0,
            package_path=package_path,
            error=str(exc),
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/models/9g_8b_thinking")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--segment", default="pre_attn", choices=("pre_attn",))
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--bucket", type=int, default=512)
    parser.add_argument("--valid-seq-len", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-root", default="")
    parser.add_argument("--package-path", default="")
    parser.add_argument("--compile", action="store_true", help="Run AOT compile before parity")
    parser.add_argument(
        "--backend",
        choices=("auto", "aot_inductor", "torch_compile"),
        default="auto",
        help="Inductor backend for compiled segment (auto prefers AOT package if present)",
    )
    parser.add_argument(
        "--require-aot",
        action="store_true",
        help="Fail if AOT packaging fails (no torch.compile fallback)",
    )
    parser.add_argument("--json-out", default="")
    args = parser.parse_args(list(argv) if argv is not None else None)

    from infinilm.compile.env import (
        piecewise_inductor_cache_root,
        piecewise_inductor_require_aot,
    )

    device = torch.device(args.device)
    cache_root = args.cache_root or piecewise_inductor_cache_root()
    valid_seq_len = args.valid_seq_len if args.valid_seq_len > 0 else args.bucket
    require_aot = args.require_aot or piecewise_inductor_require_aot()

    compiled_fn = None
    if args.compile:
        from infinilm.compile.piecewise_segments import aot_compile_piecewise_segment

        compile_summary = aot_compile_piecewise_segment(
            model_path=args.model_path,
            segment=args.segment,
            layer_idx=args.layer,
            bucket=args.bucket,
            device=device,
            cache_root=cache_root,
            valid_seq_len=valid_seq_len,
            require_aot=require_aot,
        )
        compiled_fn = compile_summary.get("compiled_fn")
        if compile_summary.get("backend") == "torch_compile":
            msg = (
                f"[segment_parity] AOT packaging failed; using torch.compile fallback "
                f"({compile_summary.get('aot_error', '')[:120]}...)"
            )
            if require_aot:
                print(msg, flush=True)
                return 1
            print(msg, flush=True)

    if args.compile and require_aot:
        args.backend = "aot_inductor"

    print(
        f"[segment_parity] segment={args.segment} L{args.layer} B{args.bucket} "
        f"valid={valid_seq_len} model={args.model_path}",
        flush=True,
    )

    result = run_segment_parity(
        model_path=args.model_path,
        segment=args.segment,
        layer_idx=args.layer,
        bucket=args.bucket,
        device=device,
        cache_root=cache_root,
        valid_seq_len=valid_seq_len,
        seed=args.seed,
        package_path=args.package_path,
        inductor_backend=args.backend,
        compiled_fn=compiled_fn,
    )

    status = "PASS" if result.passed else "FAIL"
    print(
        f"[segment_parity] {status} max_abs_diff={result.max_abs_diff:.6f} "
        f"(q={result.max_abs_diff_q:.6f} k={result.max_abs_diff_k:.6f} "
        f"v={result.max_abs_diff_v:.6f}) token_match={result.token_match} "
        f"eager_ms={result.eager_ms:.2f} aot_ms={result.aot_ms:.2f}",
        flush=True,
    )
    if result.error:
        print(f"[segment_parity] error: {result.error}", flush=True)

    summary = asdict(result)
    summary["model_path"] = args.model_path
    summary["parity_max_abs_diff_gate"] = PARITY_MAX_ABS_DIFF

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
