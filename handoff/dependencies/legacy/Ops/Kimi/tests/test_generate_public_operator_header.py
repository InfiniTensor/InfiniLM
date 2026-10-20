import importlib.util
import pathlib
import re


def _load_generator_module():
    path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "scripts"
        / "generate_public_operator_header.py"
    )
    spec = importlib.util.spec_from_file_location(
        "generate_public_operator_header_under_test", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_public_operator_header_omits_host_range_profiler(tmp_path):
    root = pathlib.Path(__file__).resolve().parents[1]
    output = tmp_path / "operator.h"
    module = _load_generator_module()

    module.generate_public_operator_header(root / "src" / "operator.h", output)

    public_header = output.read_text()
    assert '#include "host_range_profiler.h"' not in public_header
    assert (
        "static void Call(const Handle& handle, const Config& config," in public_header
    )
    for name in (
        "Make",
        "CanMake",
        "active_implementation_indices",
        "DefaultConfig",
        "DefaultImplementationIndex",
        "MakeWithDevice",
        "SelectWithDevice",
    ):
        assert re.search(rf"\b{name}\s*\([^{{;]*\)\s*;", public_header)
    assert "DispatchFunc<ActiveDevices<Key>>(\n" not in public_header
    assert "Config::Automatic()" not in public_header
    public_section = public_header.split("class Operator : public OperatorBase {", 1)[1]
    public_section = public_section.split(" protected:", 1)[0]
    assert "static Config DefaultConfig(Device::Type dev_type);" in public_section
