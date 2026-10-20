#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import pathlib


def main():
    parser = argparse.ArgumentParser(
        description="Compare two InfiniOps pytest reports."
    )
    parser.add_argument("left", type=pathlib.Path, help="First report JSON path")
    parser.add_argument("right", type=pathlib.Path, help="Second report JSON path")
    parser.add_argument(
        "--limit",
        type=int,
        default=40,
        help="Max rows to print per diff section (default: 40)",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Optional JSON path for writing the full diff report.",
    )
    args = parser.parse_args()

    diff_report = _build_diff_report(args.left, args.right)
    rendered = _render_report(diff_report, args.limit)
    print(rendered)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(diff_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"\nfull diff report written to {args.output}")


def _run_header(label, path, summary):
    env = summary.get("environment", {})
    totals = summary.get("totals", {})
    requested = ",".join(env.get("requested_devices") or ["<auto>"])

    return (
        f"{label}: {path}\n"
        f"  requested_devices={requested}\n"
        f"  torch={env.get('torch_version')}\n"
        f"  collected={totals.get('collected')} "
        f"passed={totals.get('passed')} "
        f"skipped={totals.get('skipped')} "
        f"failed={totals.get('failed')}"
    )


def _build_diff_report(left_path, right_path):
    left_summary = _load_json(left_path)
    right_summary = _load_json(right_path)
    left_details, left_detail_path = _load_details(left_path)
    right_details, right_detail_path = _load_details(right_path)

    return {
        "left": {
            "summary_path": str(left_path),
            "detail_path": str(left_detail_path),
            "detail_exists": left_detail_path.exists(),
            "summary": left_summary,
        },
        "right": {
            "summary_path": str(right_path),
            "detail_path": str(right_detail_path),
            "detail_exists": right_detail_path.exists(),
            "summary": right_summary,
        },
        "operator_diff": _build_operator_summary_diff(left_summary, right_summary),
        "case_diff": _build_case_diff(left_details, right_details),
    }


def _build_operator_summary_diff(left_summary, right_summary):
    left_ops = {_operator_key(row): row for row in left_summary.get("operators", [])}
    right_ops = {_operator_key(row): row for row in right_summary.get("operators", [])}

    left_only = sorted(set(left_ops) - set(right_ops))
    right_only = sorted(set(right_ops) - set(left_ops))
    changed = [
        key
        for key in sorted(set(left_ops) & set(right_ops))
        if _operator_payload(left_ops[key]) != _operator_payload(right_ops[key])
    ]

    return {
        "only_left_count": len(left_only),
        "only_right_count": len(right_only),
        "changed_count": len(changed),
        "only_left": [{"key": key, "row": left_ops[key]} for key in left_only],
        "only_right": [{"key": key, "row": right_ops[key]} for key in right_only],
        "changed": [
            {
                "key": key,
                "left": left_ops[key],
                "right": right_ops[key],
                "left_skip_reasons": {
                    entry["reason"]: entry["count"]
                    for entry in left_ops[key]["skip_reasons"]
                },
                "right_skip_reasons": {
                    entry["reason"]: entry["count"]
                    for entry in right_ops[key]["skip_reasons"]
                },
            }
            for key in changed
        ],
    }


def _build_case_diff(left_details, right_details):
    left_cases = {
        _case_key(record): record for record in left_details if record.get("operator")
    }
    right_cases = {
        _case_key(record): record for record in right_details if record.get("operator")
    }

    left_only = sorted(set(left_cases) - set(right_cases))
    right_only = sorted(set(right_cases) - set(left_cases))
    changed = [
        key
        for key in sorted(set(left_cases) & set(right_cases))
        if _case_payload(left_cases[key]) != _case_payload(right_cases[key])
    ]

    return {
        "only_left_count": len(left_only),
        "only_right_count": len(right_only),
        "changed_count": len(changed),
        "only_left": [{"key": key, "record": left_cases[key]} for key in left_only],
        "only_right": [{"key": key, "record": right_cases[key]} for key in right_only],
        "changed": [
            {
                "key": key,
                "left": left_cases[key],
                "right": right_cases[key],
            }
            for key in changed
        ],
    }


def _render_report(diff_report, limit):
    lines = []
    left = diff_report["left"]
    right = diff_report["right"]

    lines.append(_run_header("left", left["summary_path"], left["summary"]))
    lines.append(_run_header("right", right["summary_path"], right["summary"]))
    lines.append("")

    if not left["detail_exists"] or not right["detail_exists"]:
        missing = []

        if not left["detail_exists"]:
            missing.append(left["detail_path"])

        if not right["detail_exists"]:
            missing.append(right["detail_path"])

        lines.append("Warning")
        lines.append("  Missing detail file(s): " + ", ".join(missing))
        lines.append("  Case Diff needs both sibling `.details.jsonl` files.")
        lines.append("")

    lines.extend(_render_operator_summary_diff(diff_report["operator_diff"], limit))
    lines.append("")
    lines.extend(_render_case_diff(diff_report["case_diff"], limit))

    return "\n".join(lines)


def _render_operator_summary_diff(operator_diff, limit):
    lines = []

    lines.append("Operator Diff")
    lines.append(
        "  "
        f"only_left={operator_diff['only_left_count']} "
        f"only_right={operator_diff['only_right_count']} "
        f"changed={operator_diff['changed_count']}"
    )

    if operator_diff["only_left"]:
        lines.append("  only in left:")

        for entry in operator_diff["only_left"][:limit]:
            lines.append(f"    {entry['key']}")

    if operator_diff["only_right"]:
        lines.append("  only in right:")

        for entry in operator_diff["only_right"][:limit]:
            lines.append(f"    {entry['key']}")

    if operator_diff["changed"]:
        lines.append("  changed outcomes:")

        for entry in operator_diff["changed"][:limit]:
            lines.append(
                "    "
                f"{entry['key']}: "
                f"left={entry['left']['outcomes']} "
                f"right={entry['right']['outcomes']}"
            )

            if entry["left_skip_reasons"] != entry["right_skip_reasons"]:
                lines.append(f"      left_skip_reasons={entry['left_skip_reasons']}")
                lines.append(f"      right_skip_reasons={entry['right_skip_reasons']}")

    return lines


def _render_case_diff(case_diff, limit):
    lines = []

    lines.append("Case Diff")
    lines.append(
        "  "
        f"only_left={case_diff['only_left_count']} "
        f"only_right={case_diff['only_right_count']} "
        f"changed={case_diff['changed_count']}"
    )

    if case_diff["only_left"]:
        lines.append("  cases only in left:")

        for entry in case_diff["only_left"][:limit]:
            lines.append(f"    {entry['key']}")

    if case_diff["only_right"]:
        lines.append("  cases only in right:")

        for entry in case_diff["only_right"][:limit]:
            lines.append(f"    {entry['key']}")

    if case_diff["changed"]:
        lines.append("  same case, different result:")

        for entry in case_diff["changed"][:limit]:
            lines.append(
                "    "
                f"{entry['key']}: "
                f"left={entry['left']['outcome']} "
                f"right={entry['right']['outcome']}"
            )

            if entry["left"].get("reason") != entry["right"].get("reason"):
                lines.append(f"      left_reason={entry['left'].get('reason')}")
                lines.append(f"      right_reason={entry['right'].get('reason')}")

    return lines


def _operator_key(row):
    return (
        f"{row['module']}::{row['torch_device']}::"
        f"{row['operator']}::{row.get('aten_name')}"
    )


def _operator_payload(row):
    return {
        "cases": row["cases"],
        "outcomes": row["outcomes"],
        "skip_reasons": row["skip_reasons"],
        "implementation_indices": row["implementation_indices"],
        "dtypes": row["dtypes"],
    }


def _case_key(record):
    params = {
        key: value
        for key, value in sorted(record.get("params", {}).items())
        if key not in {"device", "rtol", "atol"}
    }
    key = {
        "module": record.get("module"),
        "operator": record.get("operator"),
        "aten_name": record.get("aten_name"),
        "implementation_index": record.get("implementation_index"),
        "params": params,
    }

    return json.dumps(key, sort_keys=True, ensure_ascii=True)


def _case_payload(record):
    return {"outcome": record.get("outcome"), "reason": record.get("reason")}


def _load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_details(summary_path):
    detail_path = summary_path.with_name(f"{summary_path.stem}.details.jsonl")

    if not detail_path.exists():
        return [], detail_path

    return (
        [
            json.loads(line)
            for line in detail_path.read_text(encoding="utf-8").splitlines()
            if line
        ],
        detail_path,
    )


if __name__ == "__main__":
    main()
