#!/usr/bin/env python3
"""严格比较 Transformers 与 InfiniLM 的真实 token-level PPL 结果。"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from _ppl_common import SCORING_METHOD, canonical_indices_sha256

RESULT_SCHEMA = "qwen3_235b_true_ppl_result/v1"
COMPARISON_SCHEMA = "qwen3_235b_true_ppl_comparison/v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class PplResult:
    path: str
    backend: str
    model: str
    corpus_manifest_sha256: str
    corpus_token_ids_sha256: str
    window_size: int
    stride: int
    scoring_method: str
    first_scored_token_index: int
    last_scored_token_index_exclusive: int
    scored_token_count: int
    scored_token_indices_sha256: str
    total_nll: float
    mean_nll: float
    ppl: float


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=[],
        help="Transformers 与 InfiniLM 结果 JSON，顺序可以互换",
    )
    parser.add_argument(
        "--max-ppl-increase-percent",
        type=float,
        default=20.0,
        help="InfiniLM 相对 Transformers 的最大 PPL 增幅，默认 20%%",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--verbose", action="store_true", help="打印完整 JSON")
    args = parser.parse_args(argv)
    args.inputs = [*args.paths, *args.inputs]
    if len(args.inputs) != 2:
        parser.error("必须提供两个结果 JSON：Transformers 与 InfiniLM")
    if (
        not math.isfinite(args.max_ppl_increase_percent)
        or args.max_ppl_increase_percent < 0
    ):
        parser.error("--max-ppl-increase-percent 必须是有限非负数")
    return args


def _required(payload: dict[str, Any], key: str, path: Path) -> Any:
    if key not in payload:
        raise ValueError(f"{path} 缺少字段 {key}")
    return payload[key]


def _sha256(payload: dict[str, Any], key: str, path: Path) -> str:
    value = str(_required(payload, key, path)).lower()
    if not SHA256_RE.fullmatch(value):
        raise ValueError(f"{path} 的 {key} 不是有效 SHA256")
    return value


def _positive_int(payload: dict[str, Any], key: str, path: Path) -> int:
    value = _required(payload, key, path)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} 的 {key} 必须是正整数")
    parsed = value
    if parsed <= 0:
        raise ValueError(f"{path} 的 {key} 必须是正整数")
    return parsed


def _nonnegative_int(payload: dict[str, Any], key: str, path: Path) -> int:
    value = _required(payload, key, path)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} 的 {key} 必须是非负整数")
    parsed = value
    if parsed < 0:
        raise ValueError(f"{path} 的 {key} 必须是非负整数")
    return parsed


def _finite(payload: dict[str, Any], key: str, path: Path) -> float:
    try:
        value = float(_required(payload, key, path))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path} 的 {key} 必须是有限数") from error
    if not math.isfinite(value):
        raise ValueError(f"{path} 的 {key} 必须是有限数")
    return value


def load_result(path: Path) -> PplResult:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"结果文件不存在：{path}") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"结果 JSON 无效：{path}: {error.msg}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"结果 JSON 必须是对象：{path}")
    if payload.get("status") != "PASS":
        raise ValueError(
            f"{path} 不是成功的 PPL 结果：status={payload.get('status')!r}"
        )
    if payload.get("schema") != RESULT_SCHEMA:
        raise ValueError(
            f"{path} schema 必须为 {RESULT_SCHEMA!r}，实际为 {payload.get('schema')!r}"
        )

    backend = str(_required(payload, "backend", path)).strip().lower()
    if backend not in {"transformers", "infinilm"}:
        raise ValueError(f"{path} backend 必须是 transformers 或 infinilm")
    model = str(_required(payload, "model", path)).strip()
    if not model:
        raise ValueError(f"{path} model 不能为空")
    window_size = _positive_int(payload, "window_size", path)
    stride = _positive_int(payload, "stride", path)
    if stride >= window_size:
        raise ValueError(f"{path} stride 必须小于 window_size")
    scoring_method = str(_required(payload, "scoring_method", path)).strip()
    if scoring_method != SCORING_METHOD:
        raise ValueError(
            f"{path} scoring_method 必须为 {SCORING_METHOD!r}"
        )
    first_index = _nonnegative_int(payload, "first_scored_token_index", path)
    last_index = _positive_int(
        payload, "last_scored_token_index_exclusive", path
    )
    scored_count = _positive_int(payload, "scored_token_count", path)
    if last_index <= first_index or last_index - first_index != scored_count:
        raise ValueError(
            f"{path} 的计分范围 [{first_index}, {last_index}) 与 "
            f"scored_token_count={scored_count} 不一致"
        )
    if first_index != 1:
        raise ValueError(f"{path} first_scored_token_index 必须为 1")
    scored_indices_hash = _sha256(
        payload, "scored_token_indices_sha256", path
    )
    expected_indices_hash = canonical_indices_sha256(
        range(first_index, last_index)
    )
    if scored_indices_hash != expected_indices_hash:
        raise ValueError(f"{path} 的 scored_token_indices_sha256 校验失败")

    total_nll = _finite(payload, "total_nll", path)
    reported_mean = _finite(payload, "mean_nll", path)
    reported_ppl = _finite(payload, "ppl", path)
    if total_nll < 0 or reported_mean < 0 or reported_ppl < 1:
        raise ValueError(f"{path} 的 NLL/PPL 超出有效范围")
    calculated_mean = total_nll / scored_count
    if calculated_mean > math.log(sys.float_info.max):
        raise ValueError(f"{path} 的 mean NLL 过大，PPL 溢出")
    calculated_ppl = math.exp(calculated_mean)
    if not math.isclose(reported_mean, calculated_mean, rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError(
            f"{path} 的 mean_nll 与 total_nll/scored_token_count 不一致"
        )
    if not math.isclose(reported_ppl, calculated_ppl, rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError(f"{path} 的 ppl 与 exp(mean_nll) 不一致")

    return PplResult(
        path=str(path),
        backend=backend,
        model=model,
        corpus_manifest_sha256=_sha256(
            payload, "corpus_manifest_sha256", path
        ),
        corpus_token_ids_sha256=_sha256(
            payload, "corpus_token_ids_sha256", path
        ),
        window_size=window_size,
        stride=stride,
        scoring_method=scoring_method,
        first_scored_token_index=first_index,
        last_scored_token_index_exclusive=last_index,
        scored_token_count=scored_count,
        scored_token_indices_sha256=scored_indices_hash,
        total_nll=total_nll,
        mean_nll=calculated_mean,
        ppl=calculated_ppl,
    )


def _ordered(results: Sequence[PplResult]) -> tuple[PplResult, PplResult]:
    by_backend = {result.backend: result for result in results}
    if len(by_backend) != 2 or set(by_backend) != {"transformers", "infinilm"}:
        raise ValueError("必须且只能包含一份 Transformers 和一份 InfiniLM 结果")
    return by_backend["transformers"], by_backend["infinilm"]


def _validate_same_workload(baseline: PplResult, candidate: PplResult) -> None:
    fields = (
        "corpus_manifest_sha256",
        "corpus_token_ids_sha256",
        "window_size",
        "stride",
        "scoring_method",
        "first_scored_token_index",
        "last_scored_token_index_exclusive",
        "scored_token_count",
        "scored_token_indices_sha256",
    )
    mismatches = [
        f"{field}: {getattr(baseline, field)!r} != {getattr(candidate, field)!r}"
        for field in fields
        if getattr(baseline, field) != getattr(candidate, field)
    ]
    if mismatches:
        raise ValueError("两侧 PPL 工作负载不一致：" + "; ".join(mismatches))


def compare(
    baseline: PplResult, candidate: PplResult, threshold_percent: float
) -> dict[str, object]:
    _validate_same_workload(baseline, candidate)
    increase_percent = (candidate.ppl / baseline.ppl - 1.0) * 100.0
    passed = increase_percent <= threshold_percent
    return {
        "schema": COMPARISON_SCHEMA,
        "status": "PASS" if passed else "FAIL",
        "baseline": asdict(baseline),
        "candidate": asdict(candidate),
        "ppl_increase_percent": increase_percent,
        "max_ppl_increase_percent": threshold_percent,
        "pass": passed,
    }


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        baseline, candidate = _ordered([load_result(path) for path in args.inputs])
        report = compare(
            baseline, candidate, float(args.max_ppl_increase_percent)
        )
        if args.json_out is not None:
            _atomic_json(args.json_out, report)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2

    print("真实 PPL 对比")
    print(
        f"Transformers：PPL={baseline.ppl:.6f}  "
        f"NLL={baseline.total_nll:.6f}  Token={baseline.scored_token_count}"
    )
    print(
        f"InfiniLM：    PPL={candidate.ppl:.6f}  "
        f"NLL={candidate.total_nll:.6f}  Token={candidate.scored_token_count}"
    )
    print(f"PPL 增幅：{report['ppl_increase_percent']:.2f}%")
    print(
        f"验收要求：增幅 <= {args.max_ppl_increase_percent:.2f}%  "
        f"结果={report['status']}"
    )
    if args.verbose:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
