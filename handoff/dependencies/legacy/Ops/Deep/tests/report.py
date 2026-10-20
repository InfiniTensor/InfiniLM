from __future__ import annotations

import json
import pathlib
import platform
import sys
import uuid
from collections import Counter, defaultdict

import pytest
import torch

from tests.utils import get_available_devices

_REPORT_FORMAT_VERSION = 1
_DETAIL_SUFFIX = ".details.jsonl"
_TEXT_SUFFIX = ".summary.txt"
_TORCH_OPS_SLOT = 8
_OPERATOR_MODULES = frozenset(
    path.stem
    for path in (pathlib.Path(__file__).resolve().parents[1] / "src" / "base").glob(
        "*.h"
    )
)


def register_reporter(config):
    report_arg = config.getoption("--report")

    if not report_arg:
        return

    reporter = _ReportPlugin(config, pathlib.Path(report_arg))
    config._infini_reporter = reporter
    config.pluginmanager.register(reporter, "infini-report")


class _ReportPlugin:
    def __init__(self, config, output_path):
        self._config = config
        self._output_path = output_path.expanduser()
        self._records_by_nodeid = {}
        self._tests_collected = 0
        self._worker_input = getattr(config, "workerinput", None)
        self._worker_id = (
            self._worker_input.get("workerid") if self._worker_input else None
        )
        self._run_id = (
            self._worker_input.get("report_run_id")
            if self._worker_input
            else uuid.uuid4().hex[:8]
        )

    def pytest_configure_node(self, node):
        node.workerinput["report_run_id"] = self._run_id

    def pytest_collection_finish(self, session):
        self._tests_collected = session.testscollected

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item, call):
        outcome = yield
        report = outcome.get_result()
        record = self._record_from_item(item, report)

        if record is not None:
            self._records_by_nodeid[record["nodeid"]] = record

    def pytest_sessionfinish(self, session, exitstatus):
        self._tests_collected = session.testscollected or self._tests_collected

        if self._worker_id is not None:
            self._write_detail_records(
                self._worker_detail_path(),
                self._sorted_records(self._records_by_nodeid.values()),
            )

            return

        records = self._sorted_records(self._records_by_nodeid.values())

        if self._xdist_enabled():
            worker_records = self._load_worker_records()

            if worker_records:
                records = worker_records

        self._write_final_reports(records, exitstatus)

    def _record_from_item(self, item, report):
        if report.when == "teardown" and not report.failed:
            return None

        if report.when == "setup" and report.passed:
            return None

        if report.when == "teardown" and report.failed:
            outcome = "failed"
        else:
            outcome = report.outcome

        context = _item_context(item)
        context["outcome"] = outcome
        context["stage"] = report.when

        reason = _report_reason(report)
        if reason:
            context["reason"] = reason

        return context

    def _xdist_enabled(self):
        numprocesses = getattr(self._config.option, "numprocesses", 0) or 0

        return numprocesses > 0

    def _detail_path(self):
        return self._output_path.with_name(f"{self._output_path.stem}{_DETAIL_SUFFIX}")

    def _text_path(self):
        return self._output_path.with_name(f"{self._output_path.stem}{_TEXT_SUFFIX}")

    def _worker_detail_path(self):
        return self._output_path.with_name(
            f"{self._output_path.stem}.{self._run_id}.{self._worker_id}{_DETAIL_SUFFIX}"
        )

    def _load_worker_records(self):
        pattern = f"{self._output_path.stem}.{self._run_id}.*{_DETAIL_SUFFIX}"
        records = []

        for path in sorted(self._output_path.parent.glob(pattern)):
            records.extend(_read_detail_records(path))

        return self._sorted_records(records)

    def _write_final_reports(self, records, exitstatus):
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        summary = _build_summary(
            config=self._config,
            output_path=self._output_path,
            run_id=self._run_id,
            tests_collected=self._tests_collected,
            exitstatus=exitstatus,
            records=records,
        )
        self._output_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self._write_detail_records(self._detail_path(), records)
        self._text_path().write_text(_render_text_summary(summary), encoding="utf-8")

        terminal = self._config.pluginmanager.get_plugin("terminalreporter")
        if terminal is not None:
            terminal.write_line(
                "report written to "
                f"{self._output_path} "
                f"(details: {self._detail_path()}, text: {self._text_path()})"
            )

    def _write_detail_records(self, path, records):
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(record, sort_keys=True) for record in records]
        payload = "\n".join(lines)

        if payload:
            payload += "\n"

        path.write_text(payload, encoding="utf-8")

    @staticmethod
    def _sorted_records(records):
        return sorted(records, key=lambda r: (r["nodeid"], r["stage"], r["outcome"]))


def _build_summary(config, output_path, run_id, tests_collected, exitstatus, records):
    outcome_counts = Counter(record["outcome"] for record in records)
    operator_rows = _build_operator_rows(records)
    collected = tests_collected or len(records)

    return {
        "format_version": _REPORT_FORMAT_VERSION,
        "run_id": run_id,
        "output_path": str(output_path),
        "invocation": {
            "args": list(config.invocation_params.args),
            "cwd": str(config.invocation_params.dir),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "available_devices": list(get_available_devices()),
            "requested_devices": list(config.getoption("--devices") or []),
        },
        "totals": {
            "collected": collected,
            "reported": len(records),
            "passed": outcome_counts.get("passed", 0),
            "skipped": outcome_counts.get("skipped", 0),
            "failed": outcome_counts.get("failed", 0),
            "exitstatus": exitstatus,
        },
        "operator_count": len(operator_rows),
        "operators": operator_rows,
    }


def _build_operator_rows(records):
    grouped = defaultdict(list)

    for record in records:
        operator = record.get("operator")

        if not operator:
            continue

        key = (
            operator,
            record.get("aten_name"),
            record.get("module"),
            record.get("torch_device"),
        )
        grouped[key].append(record)

    rows = []

    for key, group in sorted(grouped.items()):
        outcome_counts = Counter(record["outcome"] for record in group)
        skip_reasons = Counter(
            record["reason"] for record in group if record["outcome"] == "skipped"
        )
        dtypes = sorted(
            {record.get("dtype") for record in group if record.get("dtype") is not None}
        )
        implementations = sorted(
            {
                record.get("implementation_index")
                for record in group
                if record.get("implementation_index") is not None
            }
        )
        rows.append(
            {
                "operator": key[0],
                "aten_name": key[1],
                "module": key[2],
                "torch_device": key[3],
                "cases": len(group),
                "outcomes": {
                    "passed": outcome_counts.get("passed", 0),
                    "skipped": outcome_counts.get("skipped", 0),
                    "failed": outcome_counts.get("failed", 0),
                },
                "dtypes": dtypes,
                "implementation_indices": implementations,
                "skip_reasons": [
                    {"reason": reason, "count": count}
                    for reason, count in sorted(
                        skip_reasons.items(), key=lambda item: (-item[1], item[0])
                    )
                ],
            }
        )

    return rows


def _render_text_summary(summary):
    lines = []
    totals = summary["totals"]

    lines.append(f"run_id: {summary['run_id']}")
    lines.append(f"report: {summary['output_path']}")
    lines.append(
        "requested_devices: "
        + ", ".join(summary["environment"]["requested_devices"] or ["<auto>"])
    )
    lines.append(
        "available_devices: "
        + ", ".join(summary["environment"]["available_devices"] or ["<none>"])
    )
    lines.append(
        "totals: "
        f"collected={totals['collected']} "
        f"reported={totals['reported']} "
        f"passed={totals['passed']} "
        f"skipped={totals['skipped']} "
        f"failed={totals['failed']} "
        f"exitstatus={totals['exitstatus']}"
    )
    lines.append(f"operator_count: {summary['operator_count']}")
    lines.append("")
    lines.append("operators:")

    for row in summary["operators"]:
        impls = ",".join(str(i) for i in row["implementation_indices"]) or "-"
        dtypes = ",".join(row["dtypes"]) or "-"
        lines.append(
            f"{row['operator']} [{row['torch_device']}] "
            f"cases={row['cases']} "
            f"passed={row['outcomes']['passed']} "
            f"skipped={row['outcomes']['skipped']} "
            f"failed={row['outcomes']['failed']} "
            f"impls={impls} "
            f"dtypes={dtypes}"
        )

        for entry in row["skip_reasons"]:
            lines.append(f"  skip x{entry['count']}: {entry['reason']}")

    lines.append("")

    return "\n".join(lines)


def _item_context(item):
    params = getattr(getattr(item, "callspec", None), "params", {})
    op_meta = params.get("op_meta") if isinstance(params, dict) else None
    module_name = item.module.__name__.rsplit(".", 1)[-1]
    module_stem = (
        module_name.removeprefix("test_")
        if module_name.startswith("test_")
        else module_name
    )

    operator = None
    aten_name = None
    overload_name = None

    if isinstance(op_meta, dict):
        operator = op_meta.get("name")
        aten_name = op_meta.get("aten_name", operator)
        overload_name = op_meta.get("overload_name")
    elif module_name.startswith("test_") and module_stem in _OPERATOR_MODULES:
        operator = module_stem
        aten_name = module_stem

    implementation_index = params.get("implementation_index")

    if implementation_index is None and module_name == "test_torch_ops":
        implementation_index = _TORCH_OPS_SLOT

    normalized_params = {}

    for key, value in params.items():
        if key == "op_meta":
            continue

        normalized_params[key] = _normalize_value(value)

    return {
        "nodeid": item.nodeid,
        "module": item.location[0],
        "test_name": item.originalname or item.name,
        "operator": operator,
        "aten_name": aten_name,
        "overload_name": overload_name,
        "torch_device": params.get("device"),
        "dtype": _normalize_value(params.get("dtype")),
        "implementation_index": implementation_index,
        "params": normalized_params,
    }


def _normalize_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    if isinstance(value, torch.dtype):
        return str(value)

    if isinstance(value, torch.device):
        return str(value)

    if isinstance(value, pathlib.Path):
        return str(value)

    if isinstance(value, tuple):
        return [_normalize_value(v) for v in value]

    if isinstance(value, list):
        return [_normalize_value(v) for v in value]

    if isinstance(value, dict):
        return {str(k): _normalize_value(v) for k, v in value.items()}

    if isinstance(value, (set, frozenset)):
        return sorted(_normalize_value(v) for v in value)

    return repr(value)


def _report_reason(report):
    if report.passed:
        return None

    longrepr = report.longrepr

    if report.skipped and isinstance(longrepr, tuple) and len(longrepr) == 3:
        return str(longrepr[2]).strip()

    text = getattr(report, "longreprtext", "") or str(longrepr)
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    return lines[-1] if lines else None


def _read_detail_records(path):
    records = []
    text = path.read_text(encoding="utf-8")

    for line in text.splitlines():
        if line:
            records.append(json.loads(line))

    return records
