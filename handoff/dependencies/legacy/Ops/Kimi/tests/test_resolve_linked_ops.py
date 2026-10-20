import importlib.util
import json
import pathlib
import subprocess
import sys

import pytest


def _load_resolver_module():
    path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "scripts"
        / "resolve_linked_ops.py"
    )
    spec = importlib.util.spec_from_file_location("resolve_linked_ops_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_linked_config(root, *, library_extra="", binding_extra=""):
    platform = root / "torch" / "metax"
    op_dir = platform / "ops" / "silu_and_mul"
    op_dir.mkdir(parents=True)
    (platform / "vllm.yaml").write_text(
        f"python_distribution_package: vllm\nlibrary_glob: vllm/_C*.so\n{library_extra}"
    )
    (op_dir / "vllm.yaml").write_text(
        "library: vllm\n"
        "required_symbols:\n"
        "  - silu_and_mul(at::Tensor&, at::Tensor&)\n"
        f"{binding_extra}"
    )
    (op_dir / "vllm.h").write_text("// declaration\n")
    (op_dir / "vllm.cc").write_text("// definition\n")
    return platform, op_dir


def test_resolve_collects_selected_implementation_and_library(monkeypatch, tmp_path):
    module = _load_resolver_module()
    source_root = tmp_path / "linked"
    _, op_dir = _write_linked_config(source_root)
    ignored_dir = source_root / "torch" / "metax" / "ops" / "ignored"
    ignored_dir.mkdir()
    (ignored_dir / "missing.yaml").write_text(
        "library: missing\nrequired_symbols:\n  - missing()\n"
    )
    (ignored_dir / "missing.h").write_text("// ignored\n")
    (ignored_dir / "missing.cc").write_text("// ignored\n")
    library_path = tmp_path / "site-packages" / "vllm" / "_C.abi3.so"
    library_path.parent.mkdir(parents=True)
    library_path.touch()

    monkeypatch.setattr(
        module, "_locate_distribution_library", lambda config: library_path
    )
    exported = {"silu_and_mul(at::Tensor&, at::Tensor&)"}
    monkeypatch.setattr(
        module,
        "_inspect_dynamic_symbols",
        lambda library, nm, readelf, cxxfilt: (exported, exported),
    )

    output_dir = tmp_path / "generated" / "linked"
    payload = module.resolve_linked_ops(
        ["cpu;metax"],
        ["silu_and_mul"],
        source_root=source_root,
        output_dir=output_dir,
    )

    assert payload["devices"] == ["cpu", "metax"]
    assert payload["operators"] == [
        {
            "device": "metax",
            "transport": "torch",
            "implementation": "vllm",
            "library": "vllm",
            "name": "silu_and_mul",
            "required_symbols": ["silu_and_mul(at::Tensor&, at::Tensor&)"],
            "source": str((op_dir / "vllm.cc").resolve()),
        }
    ]
    assert payload["libraries"][0]["path"] == str(library_path)
    assert not payload["libraries"][0]["force_load"]
    assert json.loads((output_dir / "resolved.json").read_text()) == payload

    manifest = (output_dir / "manifest.cmake").read_text()
    assert "INFINI_OPS_LINKED_SOURCES" in manifest
    assert str(op_dir / "vllm.cc").replace("\\", "/") in manifest
    assert str(library_path).replace("\\", "/") in manifest
    assert '"torch"' in manifest
    assert "ignored" not in manifest


def test_resolve_validates_dispatcher_contract_and_force_loads_library(
    monkeypatch, tmp_path
):
    module = _load_resolver_module()
    source_root = tmp_path / "linked"
    platform = source_root / "torch" / "nvidia"
    op_dir = platform / "ops" / "gptq_marlin_repack"
    op_dir.mkdir(parents=True)
    (platform / "vllm.yaml").write_text(
        "python_distribution_package: vllm\nlibrary_glob: vllm/_C*.so\n"
    )
    schema = (
        "_C::gptq_marlin_repack(Tensor b_q_weight, Tensor perm, "
        "SymInt size_k, SymInt size_n, int num_bits, bool is_a_8bit) -> Tensor"
    )
    (op_dir / "vllm.yaml").write_text(
        f"library: vllm\noperator_schema: {schema}\ndispatch_key: CUDA\n"
    )
    (op_dir / "vllm.h").write_text("// declaration\n")
    (op_dir / "vllm.cc").write_text("// definition\n")
    library_path = tmp_path / "site-packages" / "vllm" / "_C.abi3.so"
    library_path.parent.mkdir(parents=True)
    library_path.touch()

    monkeypatch.setattr(
        module, "_locate_distribution_library", lambda config: library_path
    )
    contracts = []
    monkeypatch.setattr(
        module,
        "_verify_dispatcher_contracts",
        lambda resolved: contracts.extend(resolved),
    )
    monkeypatch.setattr(
        module,
        "_inspect_dynamic_symbols",
        lambda *args: pytest.fail("Dispatcher bindings do not inspect symbols"),
    )

    output_dir = tmp_path / "generated"
    payload = module.resolve_linked_ops(
        ["nvidia"],
        ["gptq_marlin_repack"],
        source_root=source_root,
        output_dir=output_dir,
    )

    assert len(contracts) == 1
    contract, resolved_path = contracts[0]
    assert (contract.operator_schema, contract.dispatch_key) == (schema, "CUDA")
    assert resolved_path == library_path
    assert payload["libraries"][0]["force_load"]
    assert payload["operators"] == [
        {
            "device": "nvidia",
            "transport": "torch",
            "implementation": "vllm",
            "library": "vllm",
            "name": "gptq_marlin_repack",
            "operator_schema": schema,
            "dispatch_key": "CUDA",
            "source": str((op_dir / "vllm.cc").resolve()),
        }
    ]
    manifest = (output_dir / "manifest.cmake").read_text()
    force_load_block = manifest.split(
        "set(INFINI_OPS_LINKED_FORCE_LOAD_LIBRARIES", maxsplit=1
    )[1].split(")", maxsplit=1)[0]
    assert str(library_path).replace("\\", "/") in force_load_block


def test_resolve_supports_multiple_implementations_for_one_operator(
    monkeypatch, tmp_path
):
    module = _load_resolver_module()
    source_root = tmp_path / "linked"
    platform, op_dir = _write_linked_config(source_root)
    (platform / "apex.yaml").write_text(
        "python_distribution_package: apex\nlibrary_glob: apex/_C*.so\n"
    )
    (op_dir / "apex.yaml").write_text(
        "library: apex\n"
        "required_symbols:\n"
        "  - bias_swiglu_fwd(at::Tensor&, at::Tensor&)\n"
    )
    (op_dir / "apex.h").write_text("// declaration\n")
    (op_dir / "apex.cc").write_text("// definition\n")

    libraries = {
        "apex": tmp_path / "apex" / "_C.apex.so",
        "vllm": tmp_path / "vllm" / "_C.vllm.so",
    }
    for library in libraries.values():
        library.parent.mkdir()
        library.touch()
    exports = {
        libraries["apex"]: {"bias_swiglu_fwd(at::Tensor&, at::Tensor&)"},
        libraries["vllm"]: {"silu_and_mul(at::Tensor&, at::Tensor&)"},
    }
    monkeypatch.setattr(
        module,
        "_locate_distribution_library",
        lambda config: libraries[config.name],
    )
    monkeypatch.setattr(
        module,
        "_inspect_dynamic_symbols",
        lambda library, nm, readelf, cxxfilt: (
            exports[library],
            exports[library],
        ),
    )

    payload = module.resolve_linked_ops(
        ["metax"],
        ["silu_and_mul"],
        source_root=source_root,
        output_dir=tmp_path / "generated",
    )

    assert [entry["implementation"] for entry in payload["operators"]] == [
        "apex",
        "vllm",
    ]
    assert [pathlib.Path(entry["source"]).name for entry in payload["operators"]] == [
        "apex.cc",
        "vllm.cc",
    ]


def test_resolve_filters_linked_dependencies_by_slot(monkeypatch, tmp_path):
    module = _load_resolver_module()
    source_root = tmp_path / "linked"
    platform, op_dir = _write_linked_config(source_root)
    (op_dir / "vllm.h").write_text(
        "class Operator<SiluAndMul, Device::Type::kMetax, 16> {};\n"
    )
    (platform / "unused.yaml").write_text(
        "python_distribution_package: unused\nlibrary_glob: unused/_C*.so\n"
    )
    (op_dir / "unused.yaml").write_text(
        "library: unused\nrequired_symbols:\n  - unused()\n"
    )
    (op_dir / "unused.h").write_text(
        "class Operator<SiluAndMul, Device::Type::kMetax, 17> {};\n"
    )
    (op_dir / "unused.cc").write_text("// definition\n")
    config_path = tmp_path / "ops.json"
    config_path.write_text('{"silu_and_mul": {"implementations": [16]}}\n')
    library_path = tmp_path / "vllm" / "_C.so"
    library_path.parent.mkdir()
    library_path.touch()
    located = []

    def locate(config):
        located.append(config.name)
        return library_path

    monkeypatch.setattr(module, "_locate_distribution_library", locate)
    exported = {"silu_and_mul(at::Tensor&, at::Tensor&)"}
    monkeypatch.setattr(
        module,
        "_inspect_dynamic_symbols",
        lambda *args: (exported, exported),
    )

    payload = module.resolve_linked_ops(
        ["metax"],
        config_path=config_path,
        source_root=source_root,
        output_dir=tmp_path / "generated",
    )

    assert located == ["vllm"]
    assert [op["implementation"] for op in payload["operators"]] == ["vllm"]


def test_resolve_matches_structured_implementation_descriptor(monkeypatch, tmp_path):
    module = _load_resolver_module()
    source_root = tmp_path / "linked"
    _, op_dir = _write_linked_config(source_root)
    config_path = tmp_path / "ops.json"
    config_path.write_text(
        '{"silu_and_mul": [{"path": "'
        + (op_dir / "vllm.h").relative_to(tmp_path).as_posix()
        + '", "backend": "linked"}]}\n'
    )
    library_path = tmp_path / "vllm" / "_C.so"
    library_path.parent.mkdir()
    library_path.touch()

    monkeypatch.setattr(module, "_PROJECT_DIR", tmp_path)
    monkeypatch.setattr(
        module,
        "_locate_distribution_library",
        lambda _config: library_path,
    )
    exported = {"silu_and_mul(at::Tensor&, at::Tensor&)"}
    monkeypatch.setattr(
        module,
        "_inspect_dynamic_symbols",
        lambda *args: (exported, exported),
    )

    payload = module.resolve_linked_ops(
        ["metax"],
        config_path=config_path,
        source_root=source_root,
        output_dir=tmp_path / "generated",
    )

    assert [op["implementation"] for op in payload["operators"]] == ["vllm"]


@pytest.mark.parametrize(
    ("binding", "message"),
    (
        (
            "library: vllm\n"
            "required_symbols:\n"
            "  - symbol()\n"
            "operator_schema: _C::op() -> Tensor\n"
            "dispatch_key: CUDA\n",
            "exactly one of required_symbols or operator_schema",
        ),
        (
            "library: vllm\noperator_schema: _C::op() -> Tensor\n",
            "operator_schema requires dispatch_key",
        ),
        (
            "library: vllm\nrequired_symbols:\n  - symbol()\ndispatch_key: CUDA\n",
            "dispatch_key requires operator_schema",
        ),
    ),
)
def test_resolve_requires_one_complete_binding_contract(tmp_path, binding, message):
    module = _load_resolver_module()
    source_root = tmp_path / "linked"
    _, op_dir = _write_linked_config(source_root)
    (op_dir / "vllm.yaml").write_text(binding)

    with pytest.raises(module.ResolutionError, match=message):
        module.resolve_linked_ops(
            ["metax"],
            source_root=source_root,
            output_dir=tmp_path / "generated",
        )


@pytest.mark.parametrize(
    ("library_extra", "binding_extra", "unknown_key"),
    (
        ("schema: silu_and_mul\n", "", "schema"),
        ("", "call: vllm::silu_and_mul\n", "call"),
        ("python_distribution: vllm\n", "", "python_distribution"),
        ("transport: torch\n", "", "transport"),
    ),
)
def test_resolve_rejects_unknown_yaml_keys(
    tmp_path, library_extra, binding_extra, unknown_key
):
    module = _load_resolver_module()
    source_root = tmp_path / "linked"
    _write_linked_config(
        source_root,
        library_extra=library_extra,
        binding_extra=binding_extra,
    )

    with pytest.raises(module.ResolutionError, match=f"unknown keys: {unknown_key}"):
        module.resolve_linked_ops(
            ["metax"],
            source_root=source_root,
            output_dir=tmp_path / "generated",
        )


@pytest.mark.parametrize("missing_suffix", (".h", ".cc"))
def test_resolve_requires_matching_implementation_sources(tmp_path, missing_suffix):
    module = _load_resolver_module()
    source_root = tmp_path / "linked"
    _, op_dir = _write_linked_config(source_root)
    (op_dir / f"vllm{missing_suffix}").unlink()

    with pytest.raises(
        module.ResolutionError,
        match=rf"missing sibling vllm\{missing_suffix}",
    ):
        module.resolve_linked_ops(
            ["metax"],
            source_root=source_root,
            output_dir=tmp_path / "generated",
        )


def test_resolve_rejects_both_cpp_and_cuda_sources(tmp_path):
    module = _load_resolver_module()
    source_root = tmp_path / "linked"
    _, op_dir = _write_linked_config(source_root)
    (op_dir / "vllm.cu").write_text("// duplicate definition\n")

    with pytest.raises(
        module.ResolutionError,
        match=r"both vllm\.cc and vllm\.cu are present",
    ):
        module.resolve_linked_ops(
            ["metax"],
            source_root=source_root,
            output_dir=tmp_path / "generated",
        )


def test_resolve_rejects_unknown_link_library(monkeypatch, tmp_path):
    module = _load_resolver_module()
    source_root = tmp_path / "linked"
    _write_linked_config(
        source_root,
        binding_extra="link_libraries:\n  - missing\n",
    )
    monkeypatch.setattr(
        module,
        "_locate_distribution_library",
        lambda config: tmp_path / "_C.so",
    )

    with pytest.raises(module.ResolutionError, match="unknown library 'missing'"):
        module.resolve_linked_ops(
            ["metax"],
            source_root=source_root,
            output_dir=tmp_path / "generated",
        )


def test_resolve_rejects_symbol_missing_from_either_tool(monkeypatch, tmp_path):
    module = _load_resolver_module()
    source_root = tmp_path / "linked"
    _write_linked_config(source_root)
    library_path = tmp_path / "_C.so"
    library_path.touch()

    monkeypatch.setattr(
        module, "_locate_distribution_library", lambda config: library_path
    )
    monkeypatch.setattr(
        module,
        "_inspect_dynamic_symbols",
        lambda library, nm, readelf, cxxfilt: (
            {"silu_and_mul(at::Tensor&, at::Tensor&)"},
            set(),
        ),
    )

    with pytest.raises(module.ResolutionError, match="according to readelf"):
        module.resolve_linked_ops(
            ["metax"],
            source_root=source_root,
            output_dir=tmp_path / "generated",
        )


def test_resolve_requires_exact_demangled_symbol(monkeypatch, tmp_path):
    module = _load_resolver_module()
    source_root = tmp_path / "linked"
    _write_linked_config(source_root)
    library_path = tmp_path / "_C.so"
    library_path.touch()

    monkeypatch.setattr(
        module, "_locate_distribution_library", lambda config: library_path
    )
    exported = {"vllm::silu_and_mul(at::Tensor&, at::Tensor&)"}
    monkeypatch.setattr(
        module,
        "_inspect_dynamic_symbols",
        lambda library, nm, readelf, cxxfilt: (exported, exported),
    )

    with pytest.raises(module.ResolutionError, match="according to nm and readelf"):
        module.resolve_linked_ops(
            ["metax"],
            source_root=source_root,
            output_dir=tmp_path / "generated",
        )


def test_resolve_rejects_distinct_libraries_with_same_basename(monkeypatch, tmp_path):
    module = _load_resolver_module()
    source_root = tmp_path / "linked"
    platform, _ = _write_linked_config(source_root)
    (platform / "other.yaml").write_text(
        "python_distribution_package: other\nlibrary_glob: other/_C*.so\n"
    )
    other_op_dir = platform / "ops" / "other_op"
    other_op_dir.mkdir()
    (other_op_dir / "other.yaml").write_text(
        "library: other\nrequired_symbols:\n  - other_op()\n"
    )
    (other_op_dir / "other.h").write_text("// declaration\n")
    (other_op_dir / "other.cc").write_text("// definition\n")

    vllm_library = tmp_path / "vllm" / "_C.abi3.so"
    other_library = tmp_path / "other" / "_C.abi3.so"
    vllm_library.parent.mkdir()
    other_library.parent.mkdir()
    vllm_library.touch()
    other_library.touch()
    libraries = {"vllm": vllm_library, "other": other_library}
    monkeypatch.setattr(
        module,
        "_locate_distribution_library",
        lambda config: libraries[config.name],
    )
    exports = {
        vllm_library: {"silu_and_mul(at::Tensor&, at::Tensor&)"},
        other_library: {"other_op()"},
    }
    monkeypatch.setattr(
        module,
        "_inspect_dynamic_symbols",
        lambda library, nm, readelf, cxxfilt: (exports[library], exports[library]),
    )
    output_dir = tmp_path / "generated"

    with pytest.raises(module.ResolutionError, match="share basename '_C.abi3.so'"):
        module.resolve_linked_ops(
            ["metax"], source_root=source_root, output_dir=output_dir
        )
    assert not (output_dir / "manifest.cmake").exists()


def test_resolve_rejects_symbol_exported_by_distinct_libraries(monkeypatch, tmp_path):
    module = _load_resolver_module()
    source_root = tmp_path / "linked"
    platform, _ = _write_linked_config(source_root)
    (platform / "other.yaml").write_text(
        "python_distribution_package: other\nlibrary_glob: other/_C*.so\n"
    )
    other_op_dir = platform / "ops" / "other_op"
    other_op_dir.mkdir()
    (other_op_dir / "other.yaml").write_text(
        "library: other\nrequired_symbols:\n  - other_op()\n"
    )
    (other_op_dir / "other.h").write_text("// declaration\n")
    (other_op_dir / "other.cc").write_text("// definition\n")

    vllm_library = tmp_path / "vllm" / "_C.vllm.so"
    other_library = tmp_path / "other" / "_C.other.so"
    vllm_library.parent.mkdir()
    other_library.parent.mkdir()
    vllm_library.touch()
    other_library.touch()
    libraries = {"vllm": vllm_library, "other": other_library}
    monkeypatch.setattr(
        module,
        "_locate_distribution_library",
        lambda config: libraries[config.name],
    )
    exports = {
        vllm_library: {"silu_and_mul(at::Tensor&, at::Tensor&)"},
        other_library: {
            "silu_and_mul(at::Tensor&, at::Tensor&)",
            "other_op()",
        },
    }
    monkeypatch.setattr(
        module,
        "_inspect_dynamic_symbols",
        lambda library, nm, readelf, cxxfilt: (exports[library], exports[library]),
    )

    with pytest.raises(module.ResolutionError, match="exported by multiple"):
        module.resolve_linked_ops(
            ["metax"],
            source_root=source_root,
            output_dir=tmp_path / "generated",
        )


def test_dynamic_symbol_inspection_uses_nm_readelf_and_cxxfilt(monkeypatch, tmp_path):
    module = _load_resolver_module()
    library_path = tmp_path / "_C.so"
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if command[0] == "llvm-nm":
            output = "0000000000001234 T vllm::silu_and_mul(at::Tensor&, at::Tensor&)\n"
        elif command[0] == "llvm-readelf":
            output = (
                "  12: 0000000000001234 42 FUNC GLOBAL DEFAULT 12 "
                "_ZN4vllm12silu_and_mulERN2at6TensorES3_\n"
            )
        else:
            output = "vllm::silu_and_mul(at::Tensor&, at::Tensor&)\n"
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    nm_symbols, readelf_symbols = module._inspect_dynamic_symbols(
        library_path, "llvm-nm", "llvm-readelf", "llvm-cxxfilt"
    )

    expected = {"vllm::silu_and_mul(at::Tensor&, at::Tensor&)"}
    assert nm_symbols == expected
    assert readelf_symbols == expected
    assert [call[0] for call in calls] == [
        ["llvm-nm", "-D", "--defined-only", "-C", str(library_path)],
        [
            "llvm-readelf",
            "--dyn-syms",
            "--wide",
            str(library_path),
        ],
        ["llvm-cxxfilt"],
    ]
    common = {"check": True, "capture_output": True, "text": True}
    assert calls[0][1] == common
    assert calls[1][1] == common
    assert calls[2][1] == {
        **common,
        "input": "_ZN4vllm12silu_and_mulERN2at6TensorES3_\n",
    }


def test_dispatcher_contract_validation_uses_one_isolated_process(
    monkeypatch, tmp_path
):
    module = _load_resolver_module()
    schema = "_C::op(Tensor input) -> Tensor"
    config = module.BindingConfig(
        device="nvidia",
        name="op",
        implementation="vllm",
        path=tmp_path / "vllm.yaml",
        transport="torch",
        source=tmp_path / "vllm.cc",
        library="vllm",
        required_symbols=(),
        operator_schema=schema,
        dispatch_key="CUDA",
    )
    other_schema = "other::op(Tensor input) -> Tensor"
    other_config = module.BindingConfig(
        device="nvidia",
        name="other_op",
        implementation="other",
        path=tmp_path / "other.yaml",
        transport="torch",
        source=tmp_path / "other.cc",
        library="other",
        required_symbols=(),
        operator_schema=other_schema,
        dispatch_key="CUDA",
    )
    library_path = tmp_path / "_C.so"
    other_library_path = tmp_path / "other.so"
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    module._verify_dispatcher_contracts(
        [(config, library_path), (other_config, other_library_path)]
    )

    assert len(calls) == 1
    assert calls[0][0][0:2] == [sys.executable, "-c"]
    payload = json.loads(calls[0][0][-1])
    assert payload == [
        {
            "binding_path": str(config.path),
            "library_path": str(library_path),
            "schema": schema,
            "dispatch_key": "CUDA",
        },
        {
            "binding_path": str(other_config.path),
            "library_path": str(other_library_path),
            "schema": other_schema,
            "dispatch_key": "CUDA",
        },
    ]
    assert calls[0][1] == {
        "check": True,
        "capture_output": True,
        "text": True,
    }


def test_library_glob_matches_path_segments():
    module = _load_resolver_module()

    assert module._matches_library_glob("vllm/_C.abi3.so", "vllm/_C*.so")
    assert not module._matches_library_glob("vllm/nested/_C.abi3.so", "vllm/_C*.so")


def test_locate_distribution_library_requires_one_glob_match(monkeypatch, tmp_path):
    module = _load_resolver_module()
    site_packages = tmp_path / "site-packages"
    library = site_packages / "vllm" / "_C.abi3.so"
    library.parent.mkdir(parents=True)
    library.touch()
    outside_library = tmp_path / "outside" / "_C.evil.so"
    outside_library.parent.mkdir()
    outside_library.touch()

    class FakeDistribution:
        files = (
            pathlib.PurePosixPath("vllm/__init__.py"),
            pathlib.PurePosixPath("vllm/_C.abi3.so"),
            pathlib.PurePosixPath("vllm/_C.evil.so"),
        )

        def locate_file(self, entry):
            if str(entry) == "vllm/_C.evil.so":
                return outside_library
            return site_packages / entry

        def read_text(self, filename):
            return None

    monkeypatch.setattr(
        module.importlib.metadata, "distribution", lambda name: FakeDistribution()
    )
    config = module.LibraryConfig(
        device="metax",
        name="vllm",
        path=tmp_path / "vllm.yaml",
        transport="torch",
        python_distribution_package="vllm",
        library_glob="vllm/_C*.so",
    )

    assert module._locate_distribution_library(config) == library.resolve()


def test_locate_distribution_library_falls_back_to_distribution_root(
    monkeypatch, tmp_path
):
    module = _load_resolver_module()
    site_packages = tmp_path / "site-packages"
    library = site_packages / "vllm" / "_C.abi3.so"
    library.parent.mkdir(parents=True)
    library.touch()

    class FakeDistribution:
        files = None

        def locate_file(self, entry):
            return site_packages / entry

        def read_text(self, filename):
            return None

    monkeypatch.setattr(
        module.importlib.metadata, "distribution", lambda name: FakeDistribution()
    )
    config = module.LibraryConfig(
        device="moore",
        name="vllm",
        path=tmp_path / "vllm.yaml",
        transport="torch",
        python_distribution_package="vllm",
        library_glob="vllm/_C*.so",
    )

    assert module._locate_distribution_library(config) == library.resolve()


def test_locate_distribution_library_supports_editable_install(monkeypatch, tmp_path):
    module = _load_resolver_module()
    site_packages = tmp_path / "site-packages"
    project_root = tmp_path / "vllm-project"
    library = project_root / "vllm" / "_C.abi3.so"
    library.parent.mkdir(parents=True)
    library.touch()

    class FakeDistribution:
        files = (pathlib.PurePosixPath("__editable__.vllm.pth"),)

        def locate_file(self, entry):
            return site_packages / entry

        def read_text(self, filename):
            assert filename == "direct_url.json"
            return json.dumps(
                {
                    "url": project_root.as_uri(),
                    "dir_info": {"editable": True},
                }
            )

    monkeypatch.setattr(
        module.importlib.metadata, "distribution", lambda name: FakeDistribution()
    )
    config = module.LibraryConfig(
        device="nvidia",
        name="vllm",
        path=tmp_path / "vllm.yaml",
        transport="torch",
        python_distribution_package="vllm",
        library_glob="vllm/_C*.so",
    )

    assert module._locate_distribution_library(config) == library.resolve()


def test_locate_distribution_include_matches_installed_directory(monkeypatch, tmp_path):
    module = _load_resolver_module()
    site_packages = tmp_path / "site-packages"
    include_dir = site_packages / "tvm_ffi" / "include"
    include_dir.mkdir(parents=True)

    class FakeDistribution:
        def locate_file(self, entry):
            return site_packages / entry

        def read_text(self, filename):
            return None

    monkeypatch.setattr(
        module.importlib.metadata, "distribution", lambda name: FakeDistribution()
    )
    config = module.LibraryConfig(
        device="nvidia",
        name="tvm_ffi",
        path=tmp_path / "tvm_ffi.yaml",
        transport="tvm_ffi",
        python_distribution_package="apache-tvm-ffi",
        library_glob="tvm_ffi/lib/libtvm_ffi.so",
        include_glob="tvm_ffi/include",
    )

    assert module._locate_distribution_include(config) == include_dir.resolve()


@pytest.mark.parametrize(
    "version, succeeds",
    (("0.6.6", False), ("0.6.7.post3", True), ("0.6.16", True)),
)
def test_load_distribution_enforces_version_constraint(
    monkeypatch, tmp_path, version, succeeds
):
    module = _load_resolver_module()

    class FakeDistribution:
        pass

    distribution = FakeDistribution()
    distribution.version = version
    monkeypatch.setattr(
        module.importlib.metadata, "distribution", lambda name: distribution
    )
    config = module.LibraryConfig(
        device="nvidia",
        name="sampling",
        path=tmp_path / "sampling.yaml",
        transport="tvm_ffi",
        python_distribution_package="flashinfer-jit-cache",
        python_distribution_version=">=0.6.7,<0.7",
        library_glob="sampling.so",
    )

    if succeeds:
        assert module._load_distribution(config) is distribution
    else:
        with pytest.raises(module.ResolutionError, match="does not satisfy"):
            module._load_distribution(config)


def test_load_distribution_rejects_invalid_version_constraint(monkeypatch, tmp_path):
    module = _load_resolver_module()

    class FakeDistribution:
        version = "0.6.7.post3"

    monkeypatch.setattr(
        module.importlib.metadata,
        "distribution",
        lambda name: FakeDistribution(),
    )
    config = module.LibraryConfig(
        device="nvidia",
        name="sampling",
        path=tmp_path / "sampling.yaml",
        transport="tvm_ffi",
        python_distribution_package="flashinfer-jit-cache",
        python_distribution_version="not-a-specifier",
        library_glob="sampling.so",
    )

    with pytest.raises(module.ResolutionError, match="invalid.*constraint"):
        module._load_distribution(config)


@pytest.mark.parametrize(
    "editable, url",
    ((False, None), (True, "https://example.com/vllm")),
)
def test_locate_editable_distribution_root_ignores_unsupported_metadata(
    tmp_path, editable, url
):
    module = _load_resolver_module()
    if url is None:
        url = tmp_path.as_uri()

    class FakeDistribution:
        def read_text(self, filename):
            assert filename == "direct_url.json"
            return json.dumps({"url": url, "dir_info": {"editable": editable}})

    assert module._locate_editable_distribution_root(FakeDistribution()) is None


def test_environment_selection_recognizes_ops_json(monkeypatch, tmp_path):
    module = _load_resolver_module()
    config_path = tmp_path / "ops.JSON"
    monkeypatch.setenv("INFINI_OPS_OPS", str(config_path))

    assert module._selection_from_environment(None, None) == (None, config_path)


def test_environment_selection_keeps_inline_allowlist(monkeypatch):
    module = _load_resolver_module()
    monkeypatch.setenv("INFINI_OPS_OPS", "add,gemm")

    assert module._selection_from_environment(None, None) == (["add,gemm"], None)


def test_explicit_selection_precedes_environment(monkeypatch, tmp_path):
    module = _load_resolver_module()
    env_config = tmp_path / "environment.json"
    explicit_config = tmp_path / "explicit.json"
    monkeypatch.setenv("INFINI_OPS_OPS", str(env_config))

    assert module._selection_from_environment(["add"], None) == (["add"], None)
    assert module._selection_from_environment(None, explicit_config) == (
        None,
        explicit_config,
    )


def test_resolve_tvm_ffi_cuda_source_with_link_dependency(monkeypatch, tmp_path):
    module = _load_resolver_module()
    source_root = tmp_path / "linked"
    platform = source_root / "tvm_ffi" / "nvidia"
    op_dir = platform / "ops" / "sampling"
    op_dir.mkdir(parents=True)
    (platform / "sampling.yaml").write_text(
        "python_distribution_package: flashinfer-jit-cache\n"
        "library_glob: flashinfer_jit_cache/jit_cache/sampling/sampling.so\n"
    )
    (platform / "tvm_ffi.yaml").write_text(
        "python_distribution_package: apache-tvm-ffi\n"
        "library_glob: tvm_ffi/lib/libtvm_ffi.so\n"
        "include_glob: tvm_ffi/include\n"
    )
    (op_dir / "flashinfer.yaml").write_text(
        "library: sampling\n"
        "link_libraries:\n"
        "  - tvm_ffi\n"
        "required_symbols:\n"
        "  - __tvm_ffi_softmax\n"
    )
    (op_dir / "flashinfer.h").write_text("// declaration\n")
    (op_dir / "flashinfer.cu").write_text("// definition\n")

    libraries = {
        "sampling": tmp_path / "sampling.so",
        "tvm_ffi": tmp_path / "libtvm_ffi.so",
    }
    for library in libraries.values():
        library.touch()
    include_dir = tmp_path / "include"
    include_dir.mkdir()
    monkeypatch.setattr(
        module,
        "_locate_distribution_library",
        lambda config: libraries[config.name],
    )
    monkeypatch.setattr(
        module,
        "_locate_distribution_include",
        lambda config: include_dir if config.name == "tvm_ffi" else None,
    )
    exported = {"__tvm_ffi_softmax"}
    monkeypatch.setattr(
        module,
        "_inspect_dynamic_symbols",
        lambda *args: (exported, exported),
    )

    output_dir = tmp_path / "generated"
    payload = module.resolve_linked_ops(
        ["nvidia"],
        ["sampling"],
        source_root=source_root,
        output_dir=output_dir,
    )

    assert {entry["name"] for entry in payload["libraries"]} == {
        "sampling",
        "tvm_ffi",
    }
    libraries_by_name = {entry["name"]: entry for entry in payload["libraries"]}
    assert libraries_by_name["sampling"]["force_load"] is False
    assert libraries_by_name["tvm_ffi"]["force_load"] is True
    assert payload["operators"] == [
        {
            "device": "nvidia",
            "transport": "tvm_ffi",
            "implementation": "flashinfer",
            "library": "sampling",
            "link_libraries": ["tvm_ffi"],
            "name": "sampling",
            "required_symbols": ["__tvm_ffi_softmax"],
            "source": str((op_dir / "flashinfer.cu").resolve()),
        }
    ]
    manifest = (output_dir / "manifest.cmake").read_text()
    tvm_sources = manifest.split("set(INFINI_OPS_LINKED_TVM_FFI_SOURCES", maxsplit=1)[
        1
    ].split(")", maxsplit=1)[0]
    torch_sources = manifest.split("set(INFINI_OPS_LINKED_TORCH_SOURCES", maxsplit=1)[
        1
    ].split(")", maxsplit=1)[0]
    include_dirs = manifest.split("set(INFINI_OPS_LINKED_INCLUDE_DIRS", maxsplit=1)[
        1
    ].split(")", maxsplit=1)[0]
    force_load_libraries = manifest.split(
        "set(INFINI_OPS_LINKED_FORCE_LOAD_LIBRARIES", maxsplit=1
    )[1].split(")", maxsplit=1)[0]
    assert "flashinfer.cu" in tvm_sources
    assert "flashinfer.cu" not in torch_sources
    assert str(include_dir).replace("\\", "/") in include_dirs
    assert "libtvm_ffi.so" in force_load_libraries
