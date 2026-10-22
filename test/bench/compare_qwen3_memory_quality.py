"""Compare completed control/optimized captures without loading a model or CUDA."""

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np


def compare(control_path, optimized_path, require_identical_tokens=False):
    """Validate paired artifacts and report numerical and greedy differences."""
    paths = [Path(control_path), Path(optimized_path)]
    records = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    for record, control in zip(records, (True, False)):
        if (
            not isinstance(record, dict)
            or record.get("status") != "COMPLETE"
            or record.get("framework") != "infinilm"
            or record.get("control") is not control
            or not isinstance(record.get("cases"), list)
            or not record["cases"]
        ):
            raise ValueError(
                "Expected completed, nonempty InfiniLM control/optimized captures."
            )
    digest = records[0].get("cases_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or records[1].get("cases_sha256") != digest
    ):
        raise ValueError("Capture case-file hashes must match.")
    if records[0].get("model") != records[1].get("model") or not records[0].get(
        "model"
    ):
        raise ValueError("Capture model identifiers must match.")
    count = len(records[0]["cases"])
    if len(records[1]["cases"]) != count:
        raise ValueError("Capture case counts must match.")
    rows = []
    with (
        np.load(paths[0].with_suffix(".npz"), allow_pickle=False) as left,
        np.load(paths[1].with_suffix(".npz"), allow_pickle=False) as right,
    ):
        expected = {str(i) for i in range(count)}
        if set(left.files) != expected or set(right.files) != expected:
            raise ValueError("Logit arrays must cover exactly the completed cases.")
        for i, (a, b) in enumerate(zip(records[0]["cases"], records[1]["cases"])):
            if not isinstance(a, dict) or not isinstance(b, dict):
                raise ValueError(f"Case {i} metadata must be an object.")
            for key in ("dataset", "index", "id", "token_count", "answers"):
                if key not in a or key not in b or a[key] != b[key]:
                    raise ValueError(f"Case {i} identity differs at `{key}`.")
            tokens = [a.get("token_ids"), b.get("token_ids")]
            if any(
                not isinstance(t, list)
                or not t
                or any(type(v) is not int or v < 0 for v in t)
                for t in tokens
            ):
                raise ValueError(f"Case {i} requires nonempty generated token IDs.")
            x, y = left[str(i)], right[str(i)]
            if (
                x.ndim != 1
                or x.size == 0
                or x.shape != y.shape
                or x.dtype.kind != "f"
                or y.dtype.kind != "f"
                or not np.isfinite(x).all()
                or not np.isfinite(y).all()
            ):
                raise ValueError(
                    f"Case {i} requires matching finite floating-point logit vectors."
                )
            x, y = x.astype(np.float64), y.astype(np.float64)
            if any(max(t) >= x.size for t in tokens):
                raise ValueError(f"Case {i} token IDs exceed the captured vocabulary.")
            nx, ny = float(np.linalg.norm(x)), float(np.linalg.norm(y))
            if not np.isfinite([nx, ny]).all():
                raise ValueError(f"Case {i} logit norms exceed float64 range.")
            relative_l2 = float(np.linalg.norm(x - y) / max(nx, 1e-30))
            cosine = (
                float(np.dot(x, y) / (nx * ny))
                if nx and ny
                else float(np.array_equal(x, y))
            )
            if not np.isfinite([relative_l2, cosine]).all():
                raise ValueError(f"Case {i} derived metrics must be finite.")
            same = tokens[0] == tokens[1]
            rows.append(
                {
                    "dataset": a["dataset"],
                    "index": a["index"],
                    "token_count": a["token_count"],
                    "relative_l2": relative_l2,
                    "cosine": cosine,
                    "max_abs_logit": float(np.max(np.abs(x - y))),
                    "generated_ids_equal": same,
                    "passed": relative_l2 < 0.02
                    and cosine > 0.999
                    and (same or not require_identical_tokens),
                }
            )
    return {
        "status": "PASS" if all(row["passed"] for row in rows) else "FAIL",
        "cases_sha256": digest,
        "cases": rows,
        "thresholds": {"relative_l2_below": 0.02, "cosine_above": 0.999},
        "require_identical_tokens": require_identical_tokens,
        "scope": "Paired first-generation logits and greedy IDs; not a full benchmark score or cross-engine equivalence.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--require-identical-tokens", action="store_true")
    args = parser.parse_args()
    inputs = [args.control, args.optimized]
    if args.result.resolve() in {
        path.resolve() for item in inputs for path in (item, item.with_suffix(".npz"))
    }:
        parser.error("`--result` must not overwrite a capture or its NPZ.")
    args.result.write_text('{"status": "RUNNING"}\n', encoding="utf-8")
    try:
        report = compare(args.control, args.optimized, args.require_identical_tokens)
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as error:
        report = {"status": "INVALID", "error": str(error)}
    args.result.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(report["status"])
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
