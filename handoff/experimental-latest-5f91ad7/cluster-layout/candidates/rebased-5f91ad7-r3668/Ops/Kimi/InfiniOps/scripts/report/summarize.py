#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import pathlib


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_BASE_DIR = _REPO_ROOT / "src" / "base"
_TORCH_OPS_YAML = _REPO_ROOT / "scripts" / "torch_ops.yaml"

_DISPLAY_NAMES = {
    "ascend": "Ascend",
    "cambricon": "Cambricon",
    "cpu": "CPU",
    "hygon": "Hygon",
    "iluvatar": "Iluvatar",
    "metax": "MetaX",
    "moore": "Moore",
    "nvidia": "Nvidia",
}


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Print a compact summary for one or more pytest reports, "
            "including both case-level skips and operator-level skip-only counts."
        )
    )
    parser.add_argument(
        "inputs", nargs="+", type=pathlib.Path, help="Report JSON path(s)"
    )
    parser.add_argument(
        "--show-skip-only",
        action="store_true",
        help="Also list operator names whose status is SKIPPED_ONLY.",
    )
    args = parser.parse_args()

    inventory = _load_inventory()
    first = True

    for path in args.inputs:
        payload = json.loads(path.read_text(encoding="utf-8"))

        for label, summary in _extract_summaries(path, payload):
            if not first:
                print("")
            first = False
            _print_summary(label, summary, inventory, args.show_skip_only)


def _load_inventory():
    inventory = {}

    for path in sorted(_BASE_DIR.glob("*.h")):
        inventory[path.stem] = {"operator": path.stem, "category": "native"}

    for line in _TORCH_OPS_YAML.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue

        operator = _public_op_name(stripped[2:])
        inventory.setdefault(
            operator,
            {"operator": operator, "category": "torch-generated"},
        )

    return [inventory[name] for name in sorted(inventory)]


def _public_op_name(aten_name):
    public_name = aten_name

    if public_name.startswith("_"):
        public_name = f"aten{public_name}"

    if public_name.endswith("_") and not public_name.endswith("__"):
        public_name = public_name[:-1] + "_inplace"

    return public_name


def _extract_summaries(path, payload):
    if {"left", "right", "operator_diff", "case_diff"} <= set(payload):
        return [
            (
                _summary_label(payload["left"]["summary"], suffix="left"),
                payload["left"]["summary"],
            ),
            (
                _summary_label(payload["right"]["summary"], suffix="right"),
                payload["right"]["summary"],
            ),
        ]

    return [(_summary_label(payload, fallback=path.stem), payload)]


def _summary_label(summary, fallback=None, suffix=None):
    requested = summary.get("environment", {}).get("requested_devices") or []
    key = requested[0] if requested else (fallback or "report")
    label = _DISPLAY_NAMES.get(key, key)

    if suffix is not None:
        return f"{label} ({suffix})"

    return label


def _print_summary(label, summary, inventory, show_skip_only):
    env = summary.get("environment", {})
    totals = summary.get("totals", {})
    rows = _build_rows(summary, inventory)

    tested = sum(1 for row in rows if row["status"] != "NO_PYTEST_RECORD")
    pass_gt0 = sum(1 for row in rows if row["passed"] > 0)
    any_skip = sum(1 for row in rows if row["skipped"] > 0)
    skip_only = [row for row in rows if row["status"] == "SKIPPED_ONLY"]
    failed = sum(1 for row in rows if row["status"] == "FAILED")
    no_record = sum(1 for row in rows if row["status"] == "NO_PYTEST_RECORD")

    print(f"{label}")
    print(f"  torch={env.get('torch_version')}")
    print(
        "  case totals: "
        f"collected={totals.get('collected')} "
        f"passed={totals.get('passed')} "
        f"skipped={totals.get('skipped')} "
        f"failed={totals.get('failed')}"
    )
    print(
        "  operator totals: "
        f"total={len(rows)} "
        f"tested={tested} "
        f"pass>0={pass_gt0} "
        f"any-skip={any_skip} "
        f"skip-only={len(skip_only)} "
        f"failed={failed} "
        f"no-record={no_record}"
    )

    if show_skip_only and skip_only:
        print("  skip-only operators:")
        for row in skip_only:
            print(
                "    "
                f"{row['operator']} "
                f"(cases={row['cases']}, skipped={row['skipped']}, module={row['module']})"
            )


def _build_rows(summary, inventory):
    summary_rows = _group_summary_rows(summary.get("operators", []))
    rows = []

    for item in inventory:
        summary_row = summary_rows.get(item["operator"])

        if summary_row is None:
            rows.append(
                {
                    "operator": item["operator"],
                    "status": "NO_PYTEST_RECORD",
                    "cases": 0,
                    "passed": 0,
                    "skipped": 0,
                    "failed": 0,
                    "module": "-",
                }
            )
            continue

        outcomes = summary_row["outcomes"]
        passed = outcomes.get("passed", 0)
        skipped = outcomes.get("skipped", 0)
        failed = outcomes.get("failed", 0)

        if failed > 0:
            status = "FAILED"
        elif passed > 0 and skipped > 0:
            status = "PASSED_WITH_SKIPS"
        elif passed > 0:
            status = "PASSED"
        elif skipped > 0:
            status = "SKIPPED_ONLY"
        else:
            status = "NO_PYTEST_RECORD"

        rows.append(
            {
                "operator": item["operator"],
                "status": status,
                "cases": summary_row.get("cases", 0),
                "passed": passed,
                "skipped": skipped,
                "failed": failed,
                "module": summary_row.get("module", "-"),
            }
        )

    return rows


def _group_summary_rows(summary_rows):
    grouped = {}

    for row in summary_rows:
        operator = row["operator"]
        entry = grouped.setdefault(
            operator,
            {
                "operator": operator,
                "cases": 0,
                "outcomes": {"passed": 0, "skipped": 0, "failed": 0},
                "module_set": set(),
            },
        )

        entry["cases"] += row.get("cases", 0)
        outcomes = row.get("outcomes", {})
        entry["outcomes"]["passed"] += outcomes.get("passed", 0)
        entry["outcomes"]["skipped"] += outcomes.get("skipped", 0)
        entry["outcomes"]["failed"] += outcomes.get("failed", 0)

        module_name = row.get("module")
        if module_name:
            entry["module_set"].add(module_name)

    merged = {}
    for operator, entry in grouped.items():
        modules = sorted(entry.pop("module_set"))
        entry["module"] = ",".join(modules) if modules else "-"
        merged[operator] = entry

    return merged


if __name__ == "__main__":
    main()
