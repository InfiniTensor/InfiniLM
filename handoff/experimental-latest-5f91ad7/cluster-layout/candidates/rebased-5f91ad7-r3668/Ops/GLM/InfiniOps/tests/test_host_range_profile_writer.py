import json
from types import SimpleNamespace

import pytest

from tests import host_range_profile


_CONTEXT = {
    "nodeid": "tests/test_add.py::test_case[nvidia]",
    "operator": "add",
    "backend": "nvidia",
    "phase": "warm",
}


def test_expand_summaries_writes_inclusive_and_self_rows():
    summaries = [
        {
            "range": "operator.call",
            "count": 3,
            "unit": "ns",
            "inclusive_mean": 17.5,
            "inclusive_median": 16.0,
            "self_mean": 4.5,
            "self_median": 4.0,
        }
    ]

    rows = host_range_profile.expand_summary_rows(summaries, **_CONTEXT)

    assert rows == [
        {
            **_CONTEXT,
            "range": "operator.call",
            "metric": "inclusive",
            "count": 3,
            "unit": "ns",
            "mean": 17.5,
            "median": 16.0,
        },
        {
            **_CONTEXT,
            "range": "operator.call",
            "metric": "self",
            "count": 3,
            "unit": "ns",
            "mean": 4.5,
            "median": 4.0,
        },
    ]
    assert all(tuple(row) == host_range_profile.ROW_FIELDS for row in rows)


def test_truncate_then_append_json_lines(tmp_path):
    output = tmp_path / "profile.jsonl"
    output.write_text("stale\n", encoding="utf-8")
    row = {
        **_CONTEXT,
        "range": "operator.call",
        "metric": "inclusive",
        "count": 3,
        "unit": "ns",
        "mean": 17.5,
        "median": 16.0,
    }

    host_range_profile.truncate_output(output)
    host_range_profile.append_rows(output, [row])
    host_range_profile.append_rows(output, [{**row, "phase": "cold"}])

    lines = output.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [row, {**row, "phase": "cold"}]


@pytest.mark.parametrize(
    (
        "output_path",
        "benchmark_enabled",
        "compiled",
        "xdist_workers",
        "message",
    ),
    [
        ("profile.jsonl", False, True, None, "requires --benchmark"),
        ("profile.jsonl", True, False, None, "not compiled"),
        ("profile.jsonl", True, True, "auto", "does not support pytest-xdist"),
    ],
)
def test_validate_request_rejects_invalid_profile_configuration(
    output_path, benchmark_enabled, compiled, xdist_workers, message
):
    with pytest.raises(ValueError, match=message):
        host_range_profile.validate_request(
            output_path,
            benchmark_enabled=benchmark_enabled,
            compiled=compiled,
            xdist_workers=xdist_workers,
        )


def test_validate_request_does_not_restrict_xdist_without_profile_output():
    host_range_profile.validate_request(
        None,
        benchmark_enabled=False,
        compiled=False,
        xdist_workers="logical",
    )


def test_measurement_row_converts_seconds_to_nanoseconds_and_counts_calls():
    measurement = SimpleNamespace(
        raw_times=[0.000_001, 0.000_003],
        number_per_run=4,
        mean=0.000_002,
        median=0.000_001_5,
    )

    row = host_range_profile.measurement_row(measurement, **_CONTEXT)

    assert row == {
        **_CONTEXT,
        "range": "end_to_end",
        "metric": "inclusive",
        "count": 8,
        "unit": "ns",
        "mean": 2000.0,
        "median": 1500.0,
    }


def test_backend_and_operator_names_preserve_nvidia_and_snake_case():
    module = SimpleNamespace(__name__="tests.test_causal_softmax")

    assert host_range_profile.backend_from_devices(["nvidia"], "cuda") == "nvidia"
    assert host_range_profile.operator_from_module(module) == "causal_softmax"


def test_collect_ranges_stops_after_profiled_call_raises():
    state = SimpleNamespace(active=False)

    def start():
        state.active = True

    def stop():
        state.active = False

        return []

    def fail():
        raise RuntimeError("profiled call failed")

    with pytest.raises(RuntimeError, match="profiled call failed"):
        host_range_profile.collect_ranges(start, stop, fail)

    assert state.active is False


def test_collect_ranges_synchronizes_only_outside_profile_window():
    state = SimpleNamespace(active=False)
    events = []

    def synchronize():
        assert state.active is False
        events.append("synchronize")

    def start():
        state.active = True
        events.append("start")

    def callback():
        assert state.active is True
        events.append("callback")

        return "result"

    def stop():
        assert state.active is True
        state.active = False
        events.append("stop")

        return ["summary"]

    result, summaries = host_range_profile.collect_ranges(
        start, stop, callback, synchronize=synchronize
    )

    assert result == "result"
    assert summaries == ["summary"]
    assert events == ["synchronize", "start", "callback", "stop", "synchronize"]


def test_collect_replayed_ranges_profiles_exact_measurement_call_count():
    state = SimpleNamespace(active=False, calls=0)
    measurement = SimpleNamespace(raw_times=[0.1, 0.2, 0.3], number_per_run=4)

    def start():
        state.active = True

    def stop():
        state.active = False

        return [{"range": "binding.body", "count": state.calls}]

    def callback():
        assert state.active is True
        state.calls += 1

    summaries = host_range_profile.collect_replayed_ranges(
        start, stop, callback, measurement
    )

    assert state.active is False
    assert state.calls == 12
    assert summaries == [{"range": "binding.body", "count": 12}]


@pytest.mark.parametrize(
    ("torch_device", "expected_args"),
    [
        ("cuda", ()),
        ("cuda:1", ("cuda:1",)),
        ("nvidia", ()),
        ("nvidia:1", ("cuda:1",)),
    ],
)
def test_synchronize_torch_device_uses_platform_runtime(torch_device, expected_args):
    calls = []
    torch_module = SimpleNamespace(
        cuda=SimpleNamespace(synchronize=lambda *args: calls.append(args))
    )

    host_range_profile.synchronize_torch_device(torch_module, torch_device)

    assert calls == [expected_args]


def test_synchronize_torch_device_preserves_device_object():
    calls = []
    torch_device = SimpleNamespace(type="cuda", index=1)
    torch_module = SimpleNamespace(
        cuda=SimpleNamespace(synchronize=lambda *args: calls.append(args))
    )

    host_range_profile.synchronize_torch_device(torch_module, torch_device)

    assert calls == [(torch_device,)]


def test_synchronize_torch_device_skips_cpu():
    host_range_profile.synchronize_torch_device(SimpleNamespace(), "cpu")
