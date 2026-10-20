from __future__ import annotations

import json
import pathlib


ROW_FIELDS = (
    "nodeid",
    "operator",
    "backend",
    "phase",
    "range",
    "metric",
    "count",
    "unit",
    "mean",
    "median",
)

_PLATFORM_TO_TORCH_DEVICE = {
    "nvidia": "cuda",
    "metax": "cuda",
    "iluvatar": "cuda",
    "hygon": "cuda",
    "thead": "cuda",
    "moore": "musa",
    "cambricon": "mlu",
    "ascend": "npu",
}


def validate_request(output_path, *, benchmark_enabled, compiled, xdist_workers=None):
    if not output_path:
        return

    if not benchmark_enabled:
        raise ValueError("--host-range-profile requires --benchmark")

    if xdist_workers:
        raise ValueError("host range profiling does not support pytest-xdist")

    if callable(compiled):
        compiled = compiled()

    if compiled is not True:
        raise ValueError("host-range profiling is not compiled into infini.ops")


def truncate_output(output_path):
    path = pathlib.Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")

    return path


def append_rows(output_path, rows):
    path = pathlib.Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as output:
        for row in rows:
            if set(row) != set(ROW_FIELDS):
                raise ValueError(
                    "host-range profile rows must contain exactly: "
                    + ", ".join(ROW_FIELDS)
                )

            ordered = {field: row[field] for field in ROW_FIELDS}
            output.write(json.dumps(ordered, separators=(",", ":")) + "\n")


def expand_summary_rows(summaries, *, nodeid, operator, backend, phase):
    rows = []

    for summary in summaries:
        for metric in ("inclusive", "self"):
            rows.append(
                {
                    "nodeid": nodeid,
                    "operator": operator,
                    "backend": backend,
                    "phase": phase,
                    "range": summary["range"],
                    "metric": metric,
                    "count": summary["count"],
                    "unit": summary["unit"],
                    "mean": summary[f"{metric}_mean"],
                    "median": summary[f"{metric}_median"],
                }
            )

    return rows


def measurement_row(measurement, *, nodeid, operator, backend, phase):
    return {
        "nodeid": nodeid,
        "operator": operator,
        "backend": backend,
        "phase": phase,
        "range": "end_to_end",
        "metric": "inclusive",
        "count": len(measurement.raw_times) * measurement.number_per_run,
        "unit": "ns",
        "mean": measurement.mean * 1e9,
        "median": measurement.median * 1e9,
    }


def collect_ranges(start, stop, callback, synchronize=None):
    if synchronize is not None:
        synchronize()

    try:
        start()

        try:
            result = callback()
        except BaseException:
            try:
                stop()
            except BaseException:
                pass

            raise

        try:
            summaries = stop()
        except BaseException:
            try:
                stop()
            except BaseException:
                pass

            raise

        return result, summaries
    finally:
        if synchronize is not None:
            synchronize()


def collect_replayed_ranges(start, stop, callback, measurement, synchronize=None):
    call_count = len(measurement.raw_times) * measurement.number_per_run

    def replay_calls():
        for _ in range(call_count):
            callback()

    _, summaries = collect_ranges(start, stop, replay_calls, synchronize=synchronize)

    return summaries


def synchronize_torch_device(torch_module, torch_device):
    if torch_device is None:
        return

    object_device_type = getattr(torch_device, "type", None)
    device_type = object_device_type
    if device_type is None:
        device_type = str(torch_device).split(":", maxsplit=1)[0]

    device_type = _PLATFORM_TO_TORCH_DEVICE.get(device_type, device_type)
    if device_type == "cpu":
        return

    runtime = getattr(torch_module, device_type, None)
    synchronize = getattr(runtime, "synchronize", None)
    if not callable(synchronize):
        raise RuntimeError(f"cannot synchronize PyTorch device type {device_type!r}")

    device_name = str(torch_device)
    has_explicit_index = ":" in device_name
    if object_device_type is not None:
        synchronize(torch_device)
    elif has_explicit_index:
        _, device_index = device_name.split(":", maxsplit=1)
        synchronize(f"{device_type}:{device_index}")
    else:
        synchronize()


def operator_from_module(module):
    module_name = module.__name__.rsplit(".", 1)[-1]

    if module_name.startswith("test_"):
        return module_name[len("test_") :]

    return module_name


def backend_from_devices(devices, torch_device=None):
    devices = tuple(devices or ())

    if len(devices) == 1:
        return devices[0]

    if torch_device in devices:
        return torch_device

    matching_platforms = tuple(
        device
        for device in devices
        if _PLATFORM_TO_TORCH_DEVICE.get(device) == torch_device
    )

    if len(matching_platforms) == 1:
        return matching_platforms[0]

    return torch_device or "unknown"
