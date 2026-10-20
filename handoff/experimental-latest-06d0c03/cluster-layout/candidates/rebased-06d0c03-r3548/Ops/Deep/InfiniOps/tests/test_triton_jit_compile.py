import contextlib
import importlib.util
import json
import pathlib
import sys
import types

import pytest


def _load_compile_module(monkeypatch):
    class CompilationError(Exception):
        pass

    class OutOfResources(Exception):
        pass

    fake_triton = types.ModuleType("triton")
    fake_triton.__path__ = []
    fake_triton.__version__ = "3.5.1"
    fake_triton.CompilationError = CompilationError
    fake_triton.OutOfResources = OutOfResources
    fake_triton.compiler = types.SimpleNamespace(
        ASTSource=lambda **arguments: arguments
    )
    fake_triton.runtime = types.SimpleNamespace(JITFunction=type("JITFunction", (), {}))
    fake_triton.testing = types.SimpleNamespace()

    fake_backends = types.ModuleType("triton.backends")
    fake_backends.backends = {}
    fake_backends.compiler = types.SimpleNamespace(
        GPUTarget=lambda backend, architecture, warp_size: types.SimpleNamespace(
            backend=backend, arch=architecture, warp_size=warp_size
        )
    )
    fake_triton.backends = fake_backends
    monkeypatch.setitem(sys.modules, "triton", fake_triton)
    monkeypatch.setitem(sys.modules, "triton.backends", fake_backends)

    path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "src"
        / "triton"
        / "jit"
        / "compile.py"
    )
    module_spec = importlib.util.spec_from_file_location(
        "triton_jit_compile_under_test", path
    )
    assert module_spec is not None
    assert module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    monkeypatch.setitem(sys.modules, module_spec.name, module)
    module_spec.loader.exec_module(module)

    return module, fake_triton


def test_constexprs_are_interleaved_in_ast_and_launch(monkeypatch):
    module, fake_triton = _load_compile_module(monkeypatch)
    kernel = types.SimpleNamespace(
        arg_names=["output", "BLOCK_SIZE", "value", "AXIS"],
        params=[
            types.SimpleNamespace(is_constexpr=False),
            types.SimpleNamespace(is_constexpr=True),
            types.SimpleNamespace(is_constexpr=False),
            types.SimpleNamespace(is_constexpr=True),
        ],
    )
    constexprs = {"AXIS": 1, "BLOCK_SIZE": 128}
    compilation_target = object()
    compile_calls = []

    def compile_kernel(source, *, target, options):
        compile_calls.append((source, target, options))

    fake_triton.compile = compile_kernel
    module._compile_loaded_kernel(
        kernel,
        "*fp32,i32:16",
        compilation_target,
        {
            "constexprs": constexprs,
            "num_warps": 4,
            "num_stages": 3,
        },
    )

    assert len(compile_calls) == 1
    source, target, options = compile_calls[0]
    assert target is compilation_target
    assert options == {"num_warps": 4, "num_stages": 3}

    assert source["signature"] == {
        "output": "*fp32",
        "BLOCK_SIZE": "constexpr",
        "value": "i32",
        "AXIS": "constexpr",
    }
    assert source["constexprs"] == {(1,): 128, (3,): 1}
    assert source["attrs"] == {(2,): [["tt.divisibility", 16]]}

    launch_calls = []

    class CompiledKernel:
        def __getitem__(self, grid):
            assert grid == (2, 3, 4)
            return lambda *arguments: launch_calls.append(arguments)

    def do_bench(function, *, warmup, rep, return_mode):
        assert (warmup, rep, return_mode) == (25, 100, "median")
        function()
        return 3.25

    fake_triton.testing.do_bench = do_bench
    elapsed_time = module._benchmark_candidate(
        kernel,
        CompiledKernel(),
        [0x1000, -7],
        {"grid": [2, 3, 4], "constexprs": constexprs},
        25,
        100,
    )

    assert elapsed_time == 3.25
    assert launch_calls == [(0x1000, 128, -7, 1)]


def test_ast_source_rejects_invalid_runtime_and_constexpr_inputs(monkeypatch):
    module, _ = _load_compile_module(monkeypatch)
    kernel = types.SimpleNamespace(
        arg_names=["output", "BLOCK_SIZE", "value"],
        params=[
            types.SimpleNamespace(is_constexpr=False),
            types.SimpleNamespace(is_constexpr=True),
            types.SimpleNamespace(is_constexpr=False),
        ],
    )

    with pytest.raises(ValueError, match="has 2 runtime parameters"):
        module._build_ast_source(kernel, "*fp32", {"BLOCK_SIZE": 128})

    with pytest.raises(ValueError, match="does not define constexprs: BLOCK_SIZE"):
        module._build_ast_source(kernel, "*fp32,i32", {})

    with pytest.raises(ValueError, match="unknown constexprs: UNKNOWN"):
        module._build_ast_source(
            kernel,
            "*fp32,i32",
            {"BLOCK_SIZE": 128, "UNKNOWN": 1},
        )


def test_kernel_module_uses_qualified_operator_name(monkeypatch, tmp_path):
    module, _ = _load_compile_module(monkeypatch)
    source_path = tmp_path / "jit.py"
    source_path.write_text(
        "import triton\n"
        "kernel = triton.runtime.JITFunction()\n"
        "kernel.module_name = __name__\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "_get_kernel_source_path", lambda op_name: source_path)

    kernel = module._load_kernel("qualified_op")

    assert kernel.module_name == "infini.triton.ops.qualified_op.jit"


def test_out_of_resources_does_not_publish_artifact(monkeypatch):
    module, fake_triton = _load_compile_module(monkeypatch)
    kernel = object()
    compiled_kernel = object()
    artifact_calls = []

    monkeypatch.setattr(module, "_load_kernel", lambda op_name: kernel)
    monkeypatch.setattr(module, "_create_target", lambda target: object())
    monkeypatch.setattr(
        module,
        "_compile_loaded_kernel",
        lambda kernel, signature, target, config: compiled_kernel,
    )
    monkeypatch.setattr(
        module,
        "_use_device",
        lambda target, device_id, stream: contextlib.nullcontext(),
    )

    def raise_out_of_resources(*arguments):
        raise fake_triton.OutOfResources

    monkeypatch.setattr(module, "_benchmark_candidate", raise_out_of_resources)
    monkeypatch.setattr(
        module,
        "_write_artifacts",
        lambda *arguments: artifact_calls.append(arguments),
    )
    candidate = {
        "signature": "*fp32,i32",
        "constexprs": {"BLOCK_SIZE": 128},
        "grid": [1, 1, 1],
        "num_warps": 4,
        "num_stages": 3,
        "output_prefix": "unused",
        "cache_identity": "unused",
    }

    with pytest.raises(
        RuntimeError, match="No auto-tuning candidate completed successfully"
    ):
        module.auto_tune(
            "add",
            [candidate],
            [0x1000, 7],
            0,
            25,
            100,
            {"backend": "cuda"},
            0,
        )

    assert artifact_calls == []


def test_target_and_device_context_use_triton_backend_conventions(monkeypatch):
    module, fake_triton = _load_compile_module(monkeypatch)
    target_dict = {"backend": "cuda", "architecture": "90", "warp_size": 32}

    target = module._create_target(target_dict)

    assert (target.backend, target.arch, target.warp_size) == ("cuda", 90, 32)

    events = []

    class DeviceInterface:
        @staticmethod
        def ExternalStream(stream, *, device):
            events.append(("external_stream", stream, device))
            return (stream, device)

        @staticmethod
        @contextlib.contextmanager
        def stream(stream):
            events.append(("enter_stream", stream))
            yield
            events.append(("leave_stream", stream))

    class Driver:
        def __init__(self, name, device):
            self.name = name
            self.device = device

        def get_current_device(self):
            return self.device

        def set_current_device(self, device):
            events.append(("set_device", self.name, device))
            self.device = device

        def get_device_interface(self):
            return DeviceInterface

    previous_driver = Driver("previous", 1)
    cuda_driver = Driver("cuda", 0)

    class DriverConfig:
        active = previous_driver

        @classmethod
        def set_active(cls, driver):
            events.append(("set_active", driver.name))
            cls.active = driver

    fake_triton.runtime.driver = DriverConfig
    fake_triton.backends.backends = {
        "nvidia": types.SimpleNamespace(driver=lambda: cuda_driver)
    }

    with module._use_device(target_dict, 3, 0x1234):
        assert DriverConfig.active is cuda_driver
        assert cuda_driver.device == 3

    assert DriverConfig.active is previous_driver
    assert previous_driver.device == 1
    assert ("external_stream", 0x1234, 3) in events
    assert events[-2:] == [("set_active", "previous"), ("set_device", "previous", 1)]


def test_kernel_artifact_round_trip_preserves_actual_metadata(monkeypatch, tmp_path):
    module, fake_triton = _load_compile_module(monkeypatch)
    fake_triton.compiler.make_backend = lambda target: types.SimpleNamespace(
        binary_ext="cubin"
    )
    compiled_kernel = types.SimpleNamespace(
        metadata=types.SimpleNamespace(
            name="add_kernel",
            num_warps=8,
            shared=1024,
            global_scratch_size=0,
            profile_scratch_size=0,
        ),
        asm={"cubin": b"compiled-binary"},
    )
    output_prefix = tmp_path / "kernel"

    module._write_artifacts(
        compiled_kernel, object(), output_prefix, "expected-identity"
    )
    artifact = module.read_kernel_artifact(output_prefix, "expected-identity")

    assert artifact == {
        "name": "add_kernel",
        "num_warps": 8,
        "binary": b"compiled-binary",
        "shared_memory_size": 1024,
        "global_scratch_size": 0,
        "profile_scratch_size": 0,
    }
    metadata_path = pathlib.Path(f"{output_prefix}.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["binary_ext"] == "cubin"
    assert "binary_extension" not in metadata
    assert module.read_kernel_artifact(output_prefix, "wrong-identity") is None

    metadata["num_warps"] = 0
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    assert module.read_kernel_artifact(output_prefix, "expected-identity") is None


def test_cache_path_and_auto_tuning_config_round_trip(monkeypatch, tmp_path):
    module, _ = _load_compile_module(monkeypatch)
    monkeypatch.setenv("INFINI_OPS_TRITON_CACHE_DIR", str(tmp_path))
    identity = "abc"
    config = {
        "num_warps": 4,
        "num_stages": 3,
        "constexprs": {"BLOCK_SIZE": 128},
    }

    assert module._cache_path(identity).name == "e71fa2190541574b"
    module.write_auto_tuning_config(identity, config)
    assert module.read_auto_tuning_config(identity) == config

    module._cache_path(identity, ".autotune").write_text("{", encoding="utf-8")
    assert module.read_auto_tuning_config(identity) is None


@pytest.mark.parametrize("field", ("num_warps", "num_stages"))
def test_auto_tuning_cache_rejects_zero_compile_options(monkeypatch, tmp_path, field):
    module, _ = _load_compile_module(monkeypatch)
    monkeypatch.setenv("INFINI_OPS_TRITON_CACHE_DIR", str(tmp_path))
    identity = "invalid-config"
    config = {
        "num_warps": 4,
        "num_stages": 3,
        "constexprs": {"BLOCK_SIZE": 128},
    }
    config[field] = 0

    module.write_auto_tuning_config(identity, config)

    assert module.read_auto_tuning_config(identity) is None
