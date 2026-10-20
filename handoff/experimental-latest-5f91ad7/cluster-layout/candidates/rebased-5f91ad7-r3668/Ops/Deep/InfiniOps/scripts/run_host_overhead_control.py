"""Measure InfiniOps host submission with optional host-range collection."""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import time


def measure_host_submission(
    callback,
    synchronize,
    *,
    warmup_iterations,
    iterations,
    rounds,
    before_round=None,
    after_round=None,
    timer_ns=time.perf_counter_ns,
):
    """Return per-call host timing with synchronization outside each interval."""
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative")

    if iterations <= 0:
        raise ValueError("iterations must be positive")

    if rounds <= 0:
        raise ValueError("rounds must be positive")

    for _ in range(warmup_iterations):
        callback()

    synchronize()

    samples = []

    for _ in range(rounds):
        synchronize()
        if before_round is not None:
            before_round()

        try:
            start = timer_ns()

            for _ in range(iterations):
                callback()

            elapsed = timer_ns() - start
        finally:
            try:
                if after_round is not None:
                    after_round()
            finally:
                synchronize()

        samples.append(elapsed / iterations)

    return {
        "warmup_iterations": warmup_iterations,
        "iterations_per_round": iterations,
        "rounds": rounds,
        "unit": "ns",
        "mean": statistics.fmean(samples),
        "median": statistics.median(samples),
        "samples": samples,
    }


def _nvidia_cases(device_name):
    import infini.ops as ops
    import torch

    device = torch.device(device_name)
    if device.type != "cuda":
        raise ValueError("the host-overhead control currently supports CUDA only")

    stream = torch.cuda.current_stream(device).cuda_stream

    add_input = torch.randn((13, 4), dtype=torch.float32, device=device)
    add_other = torch.randn_like(add_input)
    add_output = torch.empty_like(add_input)

    def add():
        ops.add(
            add_input,
            add_other,
            add_output,
            stream=stream,
            implementation_index=0,
        )

    gemm_a = torch.randn((4, 48, 64), dtype=torch.float32, device=device)
    gemm_b = torch.randn((4, 64, 6), dtype=torch.float32, device=device)
    gemm_y = torch.empty((4, 48, 6), dtype=torch.float32, device=device)

    def gemm():
        ops.gemm(
            gemm_a,
            gemm_b,
            None,
            1.0,
            0.0,
            False,
            False,
            gemm_y,
            stream=stream,
            implementation_index=1,
        )

    compiled = getattr(ops, "_host_range_profile_compiled", lambda: False)()
    cases = (
        (
            "add",
            {
                "shape": [13, 4],
                "dtype": "float32",
                "implementation_index": 0,
            },
            add,
        ),
        (
            "gemm",
            {
                "a_shape": [4, 48, 64],
                "b_shape": [4, 64, 6],
                "y_shape": [4, 48, 6],
                "dtype": "float32",
                "implementation_index": 1,
            },
            gemm,
        ),
    )

    return ops, torch, bool(compiled), cases


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--activate-profile", action="store_true")
    parser.add_argument("--warmup-iterations", type=int, default=2_000)
    parser.add_argument("--iterations", type=int, default=5_000)
    parser.add_argument("--rounds", type=int, default=7)

    return parser.parse_args()


def main():
    args = _parse_args()
    ops, torch, profiling_compiled, cases = _nvidia_cases(args.device)
    if args.activate_profile and not profiling_compiled:
        raise RuntimeError("cannot activate profiling in a profiling-disabled build")

    before_round = ops._host_range_profile_start if args.activate_profile else None
    after_round = ops._host_range_profile_stop if args.activate_profile else None

    results = []
    for benchmark, params, callback in cases:
        result = measure_host_submission(
            callback,
            lambda: torch.cuda.synchronize(args.device),
            warmup_iterations=args.warmup_iterations,
            iterations=args.iterations,
            rounds=args.rounds,
            before_round=before_round,
            after_round=after_round,
        )
        results.append({"benchmark": benchmark, "params": params, **result})

    report = {
        "label": args.label,
        "profiling_compiled": profiling_compiled,
        "device": args.device,
        "measurement": (
            "host submission loop timed with time.perf_counter_ns; "
            "device synchronization occurs only outside each timed interval"
        ),
        "profiling_active": args.activate_profile,
        "results": results,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
