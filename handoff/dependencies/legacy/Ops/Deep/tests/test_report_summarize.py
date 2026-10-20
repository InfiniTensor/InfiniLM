import importlib.util
import pathlib
import sys


def _load_summary_module():
    path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "scripts"
        / "report"
        / "summarize.py"
    )
    spec = importlib.util.spec_from_file_location("report_summarize_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


def test_build_rows_aggregates_multiple_summary_rows_per_operator():
    module = _load_summary_module()
    summary = {
        "operators": [
            {
                "operator": "add",
                "module": "tests/test_add.py",
                "cases": 2,
                "outcomes": {"passed": 2, "skipped": 0, "failed": 0},
            },
            {
                "operator": "add",
                "module": "tests/test_torch_ops.py",
                "cases": 3,
                "outcomes": {"passed": 1, "skipped": 2, "failed": 0},
            },
        ]
    }
    inventory = [{"operator": "add", "category": "native"}]

    rows = module._build_rows(summary, inventory)

    assert rows == [
        {
            "operator": "add",
            "status": "PASSED_WITH_SKIPS",
            "cases": 5,
            "passed": 3,
            "skipped": 2,
            "failed": 0,
            "module": "tests/test_add.py,tests/test_torch_ops.py",
        }
    ]
