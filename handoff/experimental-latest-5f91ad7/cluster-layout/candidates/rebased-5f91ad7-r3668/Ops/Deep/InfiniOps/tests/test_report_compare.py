import importlib.util
import pathlib
import sys


def _load_compare_module():
    path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "scripts"
        / "report"
        / "compare.py"
    )
    spec = importlib.util.spec_from_file_location("report_compare_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


def _operator_row(device, passed, failed=0):
    return {
        "operator": "add",
        "aten_name": "add",
        "module": "tests/test_add.py",
        "torch_device": device,
        "cases": passed + failed,
        "outcomes": {"passed": passed, "skipped": 0, "failed": failed},
        "skip_reasons": [],
        "implementation_indices": [0],
        "dtypes": ["torch.float32"],
    }


def test_operator_summary_diff_preserves_per_device_rows():
    module = _load_compare_module()
    left = {"operators": [_operator_row("cpu", 1), _operator_row("cuda", 1)]}
    right = {"operators": [_operator_row("cpu", 1), _operator_row("cuda", 0, failed=1)]}

    diff = module._build_operator_summary_diff(left, right)

    assert diff["changed_count"] == 1
    assert diff["changed"][0]["key"] == "tests/test_add.py::cuda::add::add"
