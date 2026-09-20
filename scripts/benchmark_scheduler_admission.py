#!/usr/bin/env python3
"""Benchmark KV admission using the exact requests in a manifest.

Run this script from each checkout under test.  The manifest is the only source
of request shapes, so legacy, exact-scan, and incremental runs are comparable.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

from infinilm.llm.request import InferenceRequest, RequestStatus
from infinilm.llm.sampling_params import SamplingParams
from infinilm.llm.scheduler import Scheduler

try:
    from .kv_admission_manifest import AdmissionManifest, RequestSpec, load_manifest
except ImportError:
    from kv_admission_manifest import AdmissionManifest, RequestSpec, load_manifest


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def legacy_extra_blocks(request: InferenceRequest, block_size: int) -> int:
    remaining = max(
        request.sampling_params.max_tokens - request.get_num_generated_tokens(),
        0,
    )
    return ceil_div(remaining, block_size)


def exact_extra_blocks(request: InferenceRequest, block_size: int) -> int:
    target_blocks = ceil_div(
        request.get_prompt_length() + request.sampling_params.max_tokens,
        block_size,
    )
    return max(target_blocks - len(request.block_table), 0)


def make_request(spec: RequestSpec, generated_tokens: int = 1) -> InferenceRequest:
    request = InferenceRequest(
        request_id=spec.request_id,
        prompt=spec.prompt,
        prompt_token_ids=list(spec.prompt_token_ids),
        sampling_params=SamplingParams(
            max_tokens=spec.max_tokens,
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            ignore_eos=True,
        ),
    )
    for token_id in range(generated_tokens):
        request.append_generated_token_id(token_id)
    request.status = RequestStatus.RUNNING
    return request


def build_scheduler(
    manifest: AdmissionManifest,
    running_specs: Sequence[RequestSpec] | None = None,
) -> tuple[Scheduler, list[InferenceRequest]]:
    running_specs = tuple(running_specs or manifest.wave1)
    scheduler = Scheduler(
        max_batch_size=max(len(running_specs), 1),
        num_blocks=manifest.num_blocks,
        block_size=manifest.block_size,
        max_num_batched_tokens=10**9,
        enable_prefix_caching=False,
    )
    requests = []
    for spec in running_specs:
        request = make_request(spec)
        allocation = scheduler.cache_manager.allocate_slots(spec.prompt_tokens)
        if allocation is None:
            raise RuntimeError(
                f"Manifest does not fit in cache while adding `{spec.request_id}`."
            )
        request.block_table, request.slot_mapping = allocation
        request.num_blocks = len(request.block_table)
        scheduler.complete_requests([request])
        requests.append(request)
    return scheduler, requests


def scan_reservation(
    requests: Sequence[InferenceRequest],
    block_size: int,
    calculator: Callable[[InferenceRequest, int], int],
) -> int:
    return sum(calculator(request, block_size) for request in requests)


def percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate a percentile of an empty sequence.")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@dataclass(frozen=True)
class BenchmarkResult:
    variant: str
    manifest_sha256: str
    running_requests: int
    candidate_requests: int
    legacy_reserved_blocks: int
    exact_reserved_blocks: int
    false_reserved_blocks: int
    scheduler_reserved_blocks: int | None
    admission_median_us: float
    admission_p95_us: float
    accepted_candidates: int
    expected_admission_mode: str


def benchmark(
    manifest: AdmissionManifest,
    variant: str,
    iterations: int,
    repeats: int,
) -> BenchmarkResult:
    if variant not in {"legacy", "exact-scan", "incremental"}:
        raise ValueError(f"Unknown variant: `{variant}`.")
    if iterations <= 0 or repeats <= 0:
        raise ValueError("`iterations` and `repeats` must be positive.")

    scheduler, running_requests = build_scheduler(manifest)
    candidates = [make_request(spec, generated_tokens=0) for spec in manifest.wave2]
    legacy_reserved = scan_reservation(
        running_requests, manifest.block_size, legacy_extra_blocks
    )
    exact_reserved = scan_reservation(
        running_requests, manifest.block_size, exact_extra_blocks
    )

    tracked = scheduler.get_cache_stats().get("num_reserved_decode_blocks")
    if variant == "incremental" and tracked is not None and tracked != exact_reserved:
        raise AssertionError(
            "Incremental counter disagrees with manifest state: "
            f"{tracked} != {exact_reserved}."
        )

    expected_mode = "legacy" if variant == "legacy" else "exact"
    expected_values = []
    candidate_offsets = []
    cumulative_candidate_blocks = 0
    for candidate in candidates:
        candidate_offsets.append(cumulative_candidate_blocks)
        candidate_blocks = ceil_div(
            candidate.get_prompt_length() + candidate.sampling_params.max_tokens,
            manifest.block_size,
        )
        if expected_mode == "legacy":
            required = legacy_reserved + candidate_blocks
        else:
            required = exact_reserved + candidate_blocks
        expected_values.append(
            required + scheduler.pending_kv_decode_blocks + cumulative_candidate_blocks
            <= scheduler.cache_manager.get_total_usable_blocks()
        )
        cumulative_candidate_blocks += candidate_blocks

    first_rejection = next(
        (index for index, accepted in enumerate(expected_values) if not accepted),
        len(expected_values),
    )

    # Warm the Python bytecode and queue operations before collecting samples.
    for candidate, offset in zip(candidates, candidate_offsets):
        scheduler.can_accept_request(candidate, 0, offset)

    samples_us = []
    observed = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        for _ in range(iterations):
            for candidate, offset in zip(candidates, candidate_offsets):
                observed.append(scheduler.can_accept_request(candidate, 0, offset))
        elapsed = time.perf_counter_ns() - started
        samples_us.append(elapsed / (iterations * len(candidates)) / 1_000.0)

    if any(
        actual != expected_values[index % len(expected_values)]
        for index, actual in enumerate(observed)
    ):
        raise AssertionError(f"`{variant}` admission result disagrees with reference.")

    return BenchmarkResult(
        variant=variant,
        manifest_sha256=manifest.sha256,
        running_requests=len(running_requests),
        candidate_requests=len(candidates),
        legacy_reserved_blocks=legacy_reserved,
        exact_reserved_blocks=exact_reserved,
        false_reserved_blocks=max(legacy_reserved - exact_reserved, 0),
        scheduler_reserved_blocks=tracked,
        admission_median_us=statistics.median(samples_us),
        admission_p95_us=percentile(samples_us, 0.95),
        accepted_candidates=first_rejection,
        expected_admission_mode=expected_mode,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--variant", choices=("legacy", "exact-scan", "incremental"), required=True
    )
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = benchmark(
        load_manifest(args.manifest),
        variant=args.variant,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    print(json.dumps(asdict(result), indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
