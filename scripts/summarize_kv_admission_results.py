#!/usr/bin/env python3
"""Aggregate repeated fixed-manifest scheduler and serving measurements."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Iterable

VARIANTS = ("legacy", "exact-scan", "incremental")


def percent_change(before: float, after: float) -> float:
    return (after / before - 1.0) * 100.0


def percent_reduction(before: float, after: float) -> float:
    return (1.0 - after / before) * 100.0


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_same_manifest(payloads: Iterable[dict[str, Any]]) -> str:
    hashes = {payload["manifest_sha256"] for payload in payloads}
    if len(hashes) != 1:
        raise ValueError(f"Results use different manifests: {sorted(hashes)}.")
    return hashes.pop()


def serving_summary(paths: list[Path]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in VARIANTS}
    payloads = []
    for path in paths:
        payload = load_json(path)
        payloads.append(payload)
        label = payload.get("label", path.stem)
        variant = next((name for name in VARIANTS if label.startswith(name)), None)
        if variant is None:
            raise ValueError(f"Cannot infer variant from `{path}`: `{label}`.")
        grouped[variant].append(payload)
    manifest_hash = verify_same_manifest(payloads)
    missing = [variant for variant, items in grouped.items() if not items]
    if missing:
        raise ValueError(f"Missing serving variants: {', '.join(missing)}.")

    variants = {}
    for variant, items in grouped.items():
        variants[variant] = {
            "runs": len(items),
            "successful_requests": [
                item["summary"]["overall"]["successful_requests"] for item in items
            ],
            "wave2_ttft_p50_ms": statistics.median(
                item["summary"]["wave2"]["ttft_ms"]["p50"] for item in items
            ),
            "wave2_ttft_p95_ms": statistics.median(
                item["summary"]["wave2"]["ttft_ms"]["p95"] for item in items
            ),
            "overall_output_tokens_per_second": statistics.median(
                item["summary"]["overall"]["output_tokens_per_second"] for item in items
            ),
        }

    legacy = variants["legacy"]
    exact = variants["exact-scan"]
    incremental = variants["incremental"]
    comparisons = {
        "formula_legacy_to_exact": {
            "wave2_ttft_p50_reduction_percent": percent_reduction(
                legacy["wave2_ttft_p50_ms"], exact["wave2_ttft_p50_ms"]
            ),
            "output_throughput_change_percent": percent_change(
                legacy["overall_output_tokens_per_second"],
                exact["overall_output_tokens_per_second"],
            ),
        },
        "bookkeeping_exact_to_incremental": {
            "wave2_ttft_p50_reduction_percent": percent_reduction(
                exact["wave2_ttft_p50_ms"], incremental["wave2_ttft_p50_ms"]
            ),
            "output_throughput_change_percent": percent_change(
                exact["overall_output_tokens_per_second"],
                incremental["overall_output_tokens_per_second"],
            ),
        },
        "combined_legacy_to_incremental": {
            "wave2_ttft_p50_reduction_percent": percent_reduction(
                legacy["wave2_ttft_p50_ms"], incremental["wave2_ttft_p50_ms"]
            ),
            "output_throughput_change_percent": percent_change(
                legacy["overall_output_tokens_per_second"],
                incremental["overall_output_tokens_per_second"],
            ),
        },
    }
    return {
        "manifest_sha256": manifest_hash,
        "variants": variants,
        "comparisons": comparisons,
    }


def scheduler_summary(paths: list[Path]) -> dict[str, Any]:
    payloads = [load_json(path) for path in paths]
    manifest_hash = verify_same_manifest(payloads)
    variants = {payload["variant"]: payload for payload in payloads}
    missing = [variant for variant in VARIANTS if variant not in variants]
    if missing:
        raise ValueError(f"Missing scheduler variants: {', '.join(missing)}.")
    exact_us = variants["exact-scan"]["admission_median_us"]
    incremental_us = variants["incremental"]["admission_median_us"]
    return {
        "manifest_sha256": manifest_hash,
        "variants": variants,
        "incremental_speedup_over_exact_scan": exact_us / incremental_us,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serving-dir", type=Path, required=True)
    parser.add_argument("--scheduler-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    serving_paths = sorted(args.serving_dir.glob("*.json"))
    scheduler_paths = sorted(args.scheduler_dir.glob("*.json"))
    payload = {
        "serving": serving_summary(serving_paths),
        "scheduler": scheduler_summary(scheduler_paths),
    }
    if payload["serving"]["manifest_sha256"] != payload["scheduler"]["manifest_sha256"]:
        raise ValueError("Serving and scheduler results use different manifests.")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
