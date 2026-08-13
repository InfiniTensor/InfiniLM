#!/usr/bin/env python3
"""Compare InfiniLM BF16 and W8A8 true-PPL result JSON files."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from calculate_true_ppl import _validate_same_workload, load_result


SCHEMA = "qwen3_235b_infinilm_precision_ppl_comparison/v1"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs=2, type=Path, required=True)
    parser.add_argument("--max-ppl-increase-percent", type=float, default=20.0)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.max_ppl_increase_percent < 0:
        parser.error("--max-ppl-increase-percent must be non-negative")
    return args


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        results = [load_result(path) for path in args.inputs]
        if any(result.backend != "infinilm" for result in results):
            raise ValueError("both inputs must be InfiniLM results")
        precisions: list[str] = []
        for path in args.inputs:
            payload = json.loads(path.read_text(encoding="utf-8"))
            precision = str(payload.get("precision", "")).strip().upper()
            if precision not in {"BF16", "W8A8"}:
                raise ValueError(f"{path} has invalid precision: {precision!r}")
            precisions.append(precision)
        by_precision = dict(zip(precisions, results, strict=True))
        if set(by_precision) != {"BF16", "W8A8"}:
            raise ValueError("inputs must contain one BF16 and one W8A8 result")
        baseline = by_precision["BF16"]
        candidate = by_precision["W8A8"]
        _validate_same_workload(baseline, candidate)
        increase = (candidate.ppl / baseline.ppl - 1.0) * 100.0
        threshold = float(args.max_ppl_increase_percent)
        passed = increase <= threshold
        payload = {
            "schema": SCHEMA,
            "status": "PASS" if passed else "FAIL",
            "baseline": asdict(baseline),
            "candidate": asdict(candidate),
            "ppl_increase_percent": increase,
            "max_ppl_increase_percent": threshold,
        }
        _atomic_json(args.json_out, payload)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"PPL comparison error: {error}")
        return 2

    print(f"InfiniLM BF16 PPL: {baseline.ppl:.6f}")
    print(f"InfiniLM W8A8 PPL: {candidate.ppl:.6f}")
    print(f"W8A8 PPL increase: {increase:.2f}%")
    print(f"Quality threshold: <= {threshold:.2f}%")
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
