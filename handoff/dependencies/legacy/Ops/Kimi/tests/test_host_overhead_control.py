import importlib.util
import pathlib


def _load_control_module():
    path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_host_overhead_control.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_host_overhead_control_under_test", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def test_measure_host_submission_keeps_synchronization_outside_timed_interval():
    control = _load_control_module()
    events = []
    clock = iter((100, 160, 200, 300))

    def synchronize():
        events.append("synchronize")

    def callback():
        events.append("call")

    def timer_ns():
        events.append("timer")

        return next(clock)

    def before_round():
        events.append("profile_start")

    def after_round():
        events.append("profile_stop")

    result = control.measure_host_submission(
        callback,
        synchronize,
        warmup_iterations=2,
        iterations=2,
        rounds=2,
        before_round=before_round,
        after_round=after_round,
        timer_ns=timer_ns,
    )

    assert events == [
        "call",
        "call",
        "synchronize",
        "synchronize",
        "profile_start",
        "timer",
        "call",
        "call",
        "timer",
        "profile_stop",
        "synchronize",
        "synchronize",
        "profile_start",
        "timer",
        "call",
        "call",
        "timer",
        "profile_stop",
        "synchronize",
    ]
    assert result == {
        "warmup_iterations": 2,
        "iterations_per_round": 2,
        "rounds": 2,
        "unit": "ns",
        "mean": 40.0,
        "median": 40.0,
        "samples": [30.0, 50.0],
    }


def test_measure_host_submission_rejects_empty_measurements():
    control = _load_control_module()

    for name, arguments in (
        (
            "warmup_iterations",
            {"warmup_iterations": -1, "iterations": 1, "rounds": 1},
        ),
        (
            "iterations",
            {"warmup_iterations": 1, "iterations": 0, "rounds": 1},
        ),
        (
            "rounds",
            {"warmup_iterations": 1, "iterations": 1, "rounds": 0},
        ),
    ):
        try:
            control.measure_host_submission(lambda: None, lambda: None, **arguments)
        except ValueError as error:
            assert name in str(error)
        else:
            raise AssertionError(f"expected invalid {name} to fail")
