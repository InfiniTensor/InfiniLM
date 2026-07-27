#!/usr/bin/env python3
"""Summarize an in-progress ERNIE MMMU smoke run."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

ERROR_RE = re.compile(
    r"OOM|out of memory|killed|NCCL|Xid|Traceback|RuntimeError|Exception",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "prefix",
        help="Run output prefix, or log path ending in .log",
    )
    parser.add_argument("--recent", type=int, default=5)
    return parser.parse_args()


def log_path(prefix_arg: str) -> Path:
    path = Path(prefix_arg)
    if path.suffix == ".log":
        return path
    return path.with_suffix(".log")


def parse_log(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    if not path.exists():
        return rows, [f"log not found: {path}"]
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("{") and '"subject"' in line:
            try:
                rows.append(json.loads(line))
                continue
            except json.JSONDecodeError:
                pass
        if ERROR_RE.search(line):
            errors.append(line[:1000])
    return rows, errors


def subject_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    for row in rows:
        subject = str(row.get("subject", ""))
        stats[subject]["total"] += 1
        stats[subject]["correct"] += int(row.get("ok", 0))
    return dict(stats)


def main() -> int:
    args = parse_args()
    log = log_path(args.prefix)
    prefix = log.with_suffix("")
    rows, errors = parse_log(log)
    print(f"log={log}")
    print(f"log_exists={log.exists()}")
    print(f"log_bytes={log.stat().st_size if log.exists() else 0}")
    print(f"json_exists={prefix.with_suffix('.json').exists()}")
    print(f"csv_exists={prefix.with_suffix('.csv').exists()}")
    print(f"rows={len(rows)}")
    print(
        f"subjects={json.dumps(subject_stats(rows), ensure_ascii=False, sort_keys=True)}"
    )
    print(f"errors={len(errors)}")
    for error in errors[-args.recent :]:
        print(f"error: {error}")
    print("recent_rows:")
    for row in rows[-args.recent :]:
        subset = {
            key: row.get(key)
            for key in [
                "subject",
                "id",
                "gold",
                "pred",
                "ok",
                "new_tokens",
                "hit_max_new_tokens",
                "elapsed_sec",
                "boxed_answer",
                "explicit_answer",
            ]
        }
        print(json.dumps(subset, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
