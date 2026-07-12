#!/usr/bin/env python3
"""Compare two ERNIE MMMU smoke JSON outputs."""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--left", required=True, help="Left JSON path or .tar.gz archive"
    )
    parser.add_argument(
        "--right", required=True, help="Right JSON path or .tar.gz archive"
    )
    parser.add_argument("--left-label", default="left")
    parser.add_argument("--right-label", default="right")
    parser.add_argument(
        "--left-member",
        default=None,
        help="JSON member inside left archive. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--right-member",
        default=None,
        help="JSON member inside right archive. Auto-detected if omitted.",
    )
    parser.add_argument("--output", default=None, help="Optional Markdown output path")
    return parser.parse_args()


def read_json(path_arg: str, member: str | None = None) -> dict[str, Any]:
    path = Path(path_arg)
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    if path.suffix == ".log":
        return read_log_rows(path)
    if path.name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(path, "r:gz") as archive:
            selected = select_json_member(archive, member)
            with archive.extractfile(selected) as f:
                if f is None:
                    raise FileNotFoundError(selected)
                return json.loads(f.read().decode("utf-8"))
    raise ValueError(f"Unsupported input path: {path}")


def read_log_rows(path: Path) -> dict[str, Any]:
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith("{") or '"subject"' not in line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows.append(row)
    stats = subject_stats(rows)
    return {
        "backend": "log_rows",
        "subjects": list(stats.keys()),
        "split": "",
        "prompt_style": "",
        "image_size": "",
        "tp": "",
        "tp_devices": "",
        "correct": sum(item["correct"] for item in stats.values()),
        "total": sum(item["total"] for item in stats.values()),
        "skipped": "",
        "accuracy": (
            sum(item["correct"] for item in stats.values())
            / sum(item["total"] for item in stats.values())
            if stats
            else 0.0
        ),
        "rows": rows,
    }


def select_json_member(archive: tarfile.TarFile, member: str | None) -> str:
    names = [item.name for item in archive.getmembers() if item.isfile()]
    if member:
        matches = [
            name for name in names if name == member or name.endswith("/" + member)
        ]
        if not matches:
            raise FileNotFoundError(f"Archive member not found: {member}")
        return matches[0]
    json_names = [
        name
        for name in names
        if name.endswith(".json") and "mmmu" in Path(name).name.lower()
    ]
    if len(json_names) == 1:
        return json_names[0]
    preferred = [
        name
        for name in json_names
        if Path(name).name == "ernie_vl_mmmu_cpp_50_1024.json"
    ]
    if len(preferred) == 1:
        return preferred[0]
    raise ValueError(
        "Could not auto-detect archive JSON member; pass --left-member/--right-member. "
        f"Candidates: {json_names}"
    )


def rows_by_id(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {}
    for row in data.get("rows", []):
        row_id = str(row.get("id", ""))
        if row_id:
            rows[row_id] = row
    return rows


def subject_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    for row in rows:
        subject = str(row.get("subject", ""))
        stats[subject]["total"] += 1
        stats[subject]["correct"] += int(row.get("ok", 0))
    return dict(stats)


def cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\n", "\\n").replace("|", "\\|")
    return text


def config_lines(label: str, data: dict[str, Any]) -> list[str]:
    fields = [
        "backend",
        "subjects",
        "split",
        "prompt_style",
        "image_size",
        "tp",
        "tp_devices",
        "correct",
        "total",
        "skipped",
        "accuracy",
    ]
    lines = [f"### {label}", "", "| field | value |", "|---|---|"]
    for field in fields:
        lines.append(f"| {field} | {cell(data.get(field))} |")
    return lines


def build_report(
    left: dict[str, Any],
    right: dict[str, Any],
    left_label: str,
    right_label: str,
) -> str:
    left_rows = left.get("rows", [])
    right_rows = right.get("rows", [])
    left_by_id = rows_by_id(left)
    right_by_id = rows_by_id(right)
    common_ids = sorted(left_by_id.keys() & right_by_id.keys())
    left_only = sorted(left_by_id.keys() - right_by_id.keys())
    right_only = sorted(right_by_id.keys() - left_by_id.keys())

    lines: list[str] = [
        "# MMMU Comparison",
        "",
        *config_lines(left_label, left),
        "",
        *config_lines(right_label, right),
        "",
        "## Subject Summary",
        "",
        "| subject | "
        f"{left_label} correct/total | {right_label} correct/total | delta correct |",
        "|---|---:|---:|---:|",
    ]
    left_stats = subject_stats(left_rows)
    right_stats = subject_stats(right_rows)
    for subject in sorted(set(left_stats) | set(right_stats)):
        left_stat = left_stats.get(subject, {"correct": 0, "total": 0})
        right_stat = right_stats.get(subject, {"correct": 0, "total": 0})
        lines.append(
            f"| {cell(subject)} | {left_stat['correct']}/{left_stat['total']} | "
            f"{right_stat['correct']}/{right_stat['total']} | "
            f"{right_stat['correct'] - left_stat['correct']} |"
        )

    lines.extend(
        [
            "",
            "## Row Diff",
            "",
            f"Common rows: {len(common_ids)}",
            f"Left-only rows: {len(left_only)}",
            f"Right-only rows: {len(right_only)}",
            "",
            "| id | subject | gold | "
            f"{left_label} pred/ok | {right_label} pred/ok | pred same | ok same | "
            f"{left_label} boxed/explicit | {right_label} boxed/explicit |",
            "|---|---|---|---|---|---:|---:|---|---|",
        ]
    )
    for row_id in common_ids:
        lrow = left_by_id[row_id]
        rrow = right_by_id[row_id]
        pred_same = lrow.get("pred") == rrow.get("pred")
        ok_same = str(lrow.get("ok")) == str(rrow.get("ok"))
        lines.append(
            f"| {cell(row_id)} | {cell(lrow.get('subject') or rrow.get('subject'))} | "
            f"{cell(lrow.get('gold') or rrow.get('gold'))} | "
            f"{cell(lrow.get('pred'))}/{cell(lrow.get('ok'))} | "
            f"{cell(rrow.get('pred'))}/{cell(rrow.get('ok'))} | "
            f"{int(pred_same)} | {int(ok_same)} | "
            f"{cell(lrow.get('boxed_answer'))}/{cell(lrow.get('explicit_answer'))} | "
            f"{cell(rrow.get('boxed_answer'))}/{cell(rrow.get('explicit_answer'))} |"
        )

    if left_only:
        lines.extend(["", "Left-only ids:", "", ", ".join(left_only)])
    if right_only:
        lines.extend(["", "Right-only ids:", "", ", ".join(right_only)])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    left = read_json(args.left, args.left_member)
    right = read_json(args.right, args.right_member)
    report = build_report(left, right, args.left_label, args.right_label)
    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
    else:
        sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
