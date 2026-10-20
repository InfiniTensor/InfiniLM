import pathlib
import sys

import pytest


_SCRIPTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

import ops_config  # noqa: E402


def _load(tmp_path, text):
    path = tmp_path / "ops.json"
    path.write_text(text)

    return ops_config.load_ops_config(path)


def test_loads_legacy_paths_and_slot_selections(tmp_path):
    config = _load(
        tmp_path,
        """{
  "add": "src/native/cpu/ops/add/add.h",
  "gemm": [
    "first.h",
    {"path": "custom/second.h", "backend": "custom"}
  ],
  "relu": {"implementations": "all"},
  "sampling": {"implementations": [0, 16]}
}
""",
    )

    assert config == {
        "add": {"headers": ["src/native/cpu/ops/add/add.h"], "implementations": None},
        "gemm": {
            "headers": ["first.h", {"path": "custom/second.h", "backend": "custom"}],
            "implementations": None,
        },
        "relu": {"headers": None, "implementations": None},
        "sampling": {"headers": None, "implementations": (0, 16)},
    }
    assert ops_config.selected_op_names(config) == ["add", "gemm", "relu", "sampling"]
    assert ops_config.selected_slots(config, "sampling") == (0, 16)
    assert ops_config.torch_op_names(config, ["relu", "sampling"]) == ["relu"]
    config["sampling"]["implementations"] = (8,)
    assert ops_config.torch_op_names(config, ["relu"]) == ["relu", "sampling"]


@pytest.mark.parametrize(
    ("declaration", "slot"),
    (
        ("class Operator<Add, Device::Type::kCpu> : public Add {};", 0),
        (
            "class Operator<Add, Device::Type::kNvidia,\n"
            "               16> : public Add {};",
            16,
        ),
    ),
)
def test_reads_implementation_slot(tmp_path, declaration, slot):
    header = tmp_path / "implementation.h"
    header.write_text(declaration)

    assert ops_config.implementation_slot(header) == slot


def test_reads_one_slot_declared_for_multiple_devices(tmp_path):
    header = tmp_path / "implementation.h"
    header.write_text(
        "class Operator<Add, Device::Type::kCpu, 16> : public Add {};\n"
        "class Operator<Add, Device::Type::kNvidia, 16> : public Add {};\n"
    )

    assert ops_config.implementation_slot(header) == 16


def test_rejects_multiple_slots_in_one_header(tmp_path):
    header = tmp_path / "implementation.h"
    header.write_text(
        "class Operator<Add, Device::Type::kCpu, 16> : public Add {};\n"
        "class Operator<Add, Device::Type::kNvidia, 17> : public Add {};\n"
    )

    with pytest.raises(
        ops_config.OpsConfigError,
        match="exactly one implementation slot; found 16, 17",
    ):
        ops_config.implementation_slot(header)


@pytest.mark.parametrize(
    ("text", "message"),
    (
        ('{"add": {}, "add": {"implementations": "all"}}', "duplicate key"),
        ('{"add": {}}', "missing required key"),
        ('{"add": {"slots": [0]}}', "unknown keys"),
        ('{"add": {"implementations": []}}', "must not be empty"),
        ('{"add": {"implementations": [0, 0]}}', "contain duplicates"),
        ('{"add": {"implementations": [32]}}', "between 0 and 31"),
        ('{"add": {"implementations": [true]}}', "between 0 and 31"),
        ('{"add": ["first.h", 1]}', "paths or structured descriptors"),
        ('{"add": [{"path": "x.h"}]}', "non-empty string 'backend'"),
        ('{"add": [{"backend": "custom"}]}', "non-empty string 'path'"),
        (
            '{"add": [{"path": "x.h", "backend": "custom", "slot": 1}]}',
            "unknown keys",
        ),
    ),
)
def test_rejects_invalid_config(tmp_path, text, message):
    with pytest.raises(ops_config.OpsConfigError, match=message):
        _load(tmp_path, text)
