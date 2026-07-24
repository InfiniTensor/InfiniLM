import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read_source(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


class InfiniCorePythonContractsTest(unittest.TestCase):
    def test_build_installs_one_shared_runtime_for_both_extensions(self) -> None:
        xmake = read_source("xmake.lua")
        setup = read_source("setup.py")

        runtime_start = xmake.index('target("infinicore_runtime")')
        runtime_block = xmake[
            runtime_start : xmake.index("target_end()", runtime_start)
        ]
        self.assertIn('set_kind("shared")', runtime_block)
        self.assertIn('set_installdir("python/infinicore")', runtime_block)
        modern_libraries = 'add_links("infiniops", "infiniccl", "infinirt"'
        self.assertIn(modern_libraries, runtime_block)

        self.assertIn('target("_infinicore")', xmake)
        self.assertIn('add_deps("infinicore_runtime")', xmake)
        self.assertIn("csrc/infinicore/src/pybind11/infinicore.cc", xmake)
        self.assertIn("csrc/infinicore/src/pybind11/from_list.cc", xmake)
        self.assertIn('set_installdir("python/infinicore")', xmake)
        self.assertIn(
            'for target in ("infinicore_runtime", "_infinicore", "_infinilm"):',
            setup,
        )
        self.assertIn('["xmake", "build", target]', setup)
        self.assertIn('["xmake", "install", target]', setup)

        for target in ("_infinicore", "_infinilm"):
            start = xmake.index(f'target("{target}")')
            block = xmake[start : xmake.index("target_end()", start)]
            self.assertNotIn(modern_libraries, block)
            self.assertIn('add_deps("infinicore_runtime")', block)
            self.assertNotIn('add_linkgroups("infinicore_runtime"', block)

        infinicore_start = xmake.index('target("_infinicore")')
        infinicore_block = xmake[
            infinicore_start : xmake.index("target_end()", infinicore_start)
        ]
        infinilm_start = xmake.index('target("_infinilm")')
        infinilm_block = xmake[
            infinilm_start : xmake.index("target_end()", infinilm_start)
        ]
        self.assertIn('add_rpathdirs("$ORIGIN")', runtime_block)
        self.assertNotIn("add_rpathdirs(INFINI_ROOT", runtime_block)
        self.assertIn('add_rpathdirs("$ORIGIN")', infinicore_block)
        self.assertIn('add_rpathdirs("$ORIGIN/../../infinicore/lib")', infinilm_block)

    def test_wheel_contains_native_artifacts_and_is_platform_specific(self) -> None:
        setup = read_source("setup.py")

        self.assertIn("class BinaryDistribution(Distribution):", setup)
        self.assertIn("def has_ext_modules(self):", setup)
        self.assertIn("distclass=BinaryDistribution", setup)
        self.assertIn('"infinicore.lib": INFINICORE_NATIVE_ARTIFACTS', setup)
        self.assertIn('"infinilm.lib": INFINILM_EXTENSION_ARTIFACTS', setup)
        for extension in ("_infinicore", "_infinilm"):
            self.assertIn(f'"{extension}*.so"', setup)
            self.assertIn(f'"{extension}*.pyd"', setup)
        for library in (
            "infinicore_runtime",
            "infiniops",
            "infiniccl",
            "infinirt",
        ):
            self.assertIn(f'"lib{library}.so"', setup)
            self.assertIn(f'"lib{library}.so.*"', setup)

    def test_wheel_stages_modern_infini_runtime_dependencies(self) -> None:
        setup = read_source("setup.py")

        self.assertIn(
            'INFINI_LIBRARY_NAMES = ("infiniops", "infiniccl", "infinirt")',
            setup,
        )
        self.assertIn("def stage_runtime_dependencies():", setup)
        self.assertIn('destination = PROJECT_ROOT / "python/infinicore/lib"', setup)
        self.assertIn('infini_root / "lib"', setup)
        self.assertIn('infini_root / "lib64"', setup)
        self.assertIn("raise FileNotFoundError", setup)
        self.assertIn("stage_runtime_dependencies()", setup)
        build_start = setup.index("def build_cpp_module():")
        stage_call = setup.index("stage_runtime_dependencies()", build_start)
        xmake_build = setup.index('subprocess.run(["xmake", "build"', build_start)
        self.assertLess(stage_call, xmake_build)

    def test_wheel_includes_all_python_packages(self) -> None:
        python_root = ROOT / "python"
        setup = read_source("setup.py")
        missing_package_markers = sorted(
            str(directory.relative_to(python_root))
            for directory in python_root.rglob("*")
            if directory.is_dir()
            and any(directory.glob("*.py"))
            and not (directory / "__init__.py").is_file()
        )

        self.assertIn('packages=find_packages(where="python")', setup)
        self.assertEqual(missing_package_markers, [])

    def test_wheel_declares_its_eager_import_dependencies(self) -> None:
        metadata = read_source("pyproject.toml")
        dependencies = metadata[
            metadata.index("dependencies = [") : metadata.index(
                "]", metadata.index("dependencies = [")
            )
        ]
        for package in (
            "janus",
            "numpy",
            "Pillow",
            "safetensors",
            "tokenizers",
            "torch",
            "tqdm",
            "transformers",
            "typing-extensions",
            "xxhash",
        ):
            self.assertIn(f'"{package}"', dependencies)

    def test_device_binding_uses_native_infini_rt_types(self) -> None:
        source = read_source("csrc/infinicore/src/pybind11/device.hpp")

        for native_name in (
            "kCpu",
            "kNvidia",
            "kCambricon",
            "kAscend",
            "kMetax",
            "kMoore",
            "kIluvatar",
            "kHygon",
        ):
            self.assertIn(f"Device::Type::{native_name}", source)
        for legacy_name in ("::CPU", "::NVIDIA", "::QY", "::KUNLUN", "::ALI"):
            self.assertNotIn(legacy_name, source)
        self.assertIn("&Device::type", source)
        self.assertIn("&Device::index", source)
        self.assertIn("&Device::ToString", source)

    def test_dtype_binding_matches_native_infini_rt_set(self) -> None:
        source = read_source("csrc/infinicore/src/pybind11/dtype.hpp")
        native_names = (
            "kInt8",
            "kInt16",
            "kInt32",
            "kInt64",
            "kUInt8",
            "kUInt16",
            "kUInt32",
            "kUInt64",
            "kFloat16",
            "kBFloat16",
            "kFloat32",
            "kFloat64",
        )
        for native_name in native_names:
            self.assertIn(f"DataType::{native_name}", source)
        for removed_name in ("BOOL", "BYTE", "F8", "C16", "C32", "C64", "C128"):
            self.assertNotIn(f"DataType::{removed_name}", source)

    def test_from_list_and_stream_binding_do_not_use_legacy_runtime(self) -> None:
        from_list = read_source("csrc/infinicore/src/pybind11/from_list.cc")
        context = read_source("csrc/infinicore/src/pybind11/context.hpp")
        event = read_source("csrc/infinicore/src/pybind11/device_event.hpp")

        for native_name in (
            "kInt8",
            "kInt16",
            "kInt32",
            "kInt64",
            "kUInt8",
            "kUInt16",
            "kUInt32",
            "kUInt64",
            "kFloat16",
            "kBFloat16",
            "kFloat32",
            "kFloat64",
        ):
            self.assertIn(f"case DataType::{native_name}:", from_list)
        self.assertNotIn("infinirt", from_list + context + event)
        self.assertIn("reinterpret_cast<std::uintptr_t>(getStream())", context)

    def test_python_package_exposes_only_the_migrated_surface(self) -> None:
        package = ROOT / "python" / "infinicore"
        for relative_path in (
            "__init__.py",
            "context.py",
            "device.py",
            "dtype.py",
            "tensor.py",
            "utils.py",
            "lib/__init__.py",
            "nn/__init__.py",
            "nn/parameter.py",
            "nn/modules/module.py",
        ):
            self.assertTrue((package / relative_path).is_file(), relative_path)

        init = (package / "__init__.py").read_text(encoding="utf-8")
        for public_name in (
            "Tensor",
            "device",
            "dtype",
            "empty",
            "from_list",
            "from_torch",
            "cat",
            "sync_device",
            "sync_stream",
            "cancel_graph_recording",
        ):
            self.assertIn(f'"{public_name}"', init)
        context = (package / "context.py").read_text(encoding="utf-8")
        self.assertIn("def cancel_graph_recording():", context)
        self.assertIn("_infinicore.cancel_graph_recording()", context)
        for removed_name in ("infiniStatus_t", "QY", "KUNLUN", "ALI"):
            self.assertNotIn(removed_name, init)

    def test_legacy_inference_runtime_and_device_aliases_are_removed(self) -> None:
        obsolete_paths = (
            "src",
            "include/infinicore_infer.h",
            "include/infinicore_infer",
            "scripts/libinfinicore_infer",
            "scripts/deepseek.py",
            "scripts/infer_task.py",
            "scripts/jiuge.py",
            "scripts/jiuge_awq.py",
            "scripts/jiuge_gptq.py",
            "scripts/jiuge_ppl.py",
            "scripts/kvcache_pool.py",
            "scripts/launch_server.py",
            "scripts/qwen3vl.py",
            "scripts/test_ceval.py",
        )
        for relative_path in obsolete_paths:
            self.assertFalse((ROOT / relative_path).exists(), relative_path)

        base_config = read_source("python/infinilm/base_config.py")
        benchmark = read_source("test/bench/backends/infinilm.py")
        readme = read_source("README.md")
        for legacy_device in ("qy", "kunlun", "ali"):
            self.assertNotIn(f'"{legacy_device}"', base_config)
            self.assertNotIn(f'"{legacy_device}"', benchmark)
            self.assertNotIn(f"--{legacy_device}", readme)
        self.assertIn("raise ValueError", base_config)
        self.assertNotIn("--attn=flash-attn", readme)

        llama_utils = read_source("test/models/llama/utils.py")
        for legacy_dtype in (
            "DataType.F32",
            "DataType.F16",
            "DataType.BF16",
            "DataType.I8",
            "DataType.I16",
            "DataType.I32",
            "DataType.I64",
            "DataType.U8",
            "DataType.BOOL",
        ):
            self.assertNotIn(legacy_dtype, llama_utils)
        intermediate_test = read_source(
            "test/models/llama/test_intermediate_validation.py"
        )
        self.assertIn("from infinilm.lib import _infinilm", intermediate_test)
        self.assertNotIn("\n    import _infinilm", intermediate_test)
        self.assertIn("from infinicore.ops import add, matmul", intermediate_test)
        self.assertNotIn("from infinicore.ops.", intermediate_test)

    def test_device_sentinel_is_not_public_or_indexable(self) -> None:
        binding = read_source("csrc/infinicore/src/pybind11/device.hpp")
        self.assertNotIn('.value("COUNT"', binding)

        context = read_source("csrc/infinicore/src/context/context_impl.cc")
        self.assertIn("type_index >= runtime_table_.size()", context)
        self.assertIn('throw std::invalid_argument("invalid device type")', context)

    def test_from_torch_disambiguates_cuda_compatible_devices(self) -> None:
        source = read_source("python/infinicore/tensor.py")

        self.assertIn("def from_torch(torch_tensor, *, device=None):", source)
        self.assertIn('owner.device.type == "cuda"', source)
        self.assertIn("get_device()", source)
        self.assertIn("Device(device)", source)
        self.assertIn("pass device=", source)

    def test_from_torch_returns_an_owning_synchronized_copy(self) -> None:
        source = read_source("python/infinicore/tensor.py")
        from_torch = source[
            source.index("def from_torch(") : source.index("def from_numpy(")
        ]

        self.assertIn("torch.cuda.synchronize(owner.device)", from_torch)
        self.assertIn("borrowed = Tensor(", from_torch)
        self.assertIn("result = empty(", from_torch)
        self.assertIn("result.copy_(borrowed)", from_torch)
        self.assertIn("sync_stream()", from_torch)
        self.assertNotIn("return Tensor(\n        _infinicore.from_blob", from_torch)

    def test_from_list_reads_uint64_as_unsigned(self) -> None:
        source = read_source("csrc/infinicore/src/pybind11/from_list.cc")

        self.assertIn("PyLong_AsUnsignedLongLong", source)
        self.assertIn("read_pyuint64", source)
        self.assertIn("write_dtype_native<uint64_t>", source)

        utils = read_source("python/infinicore/utils.py")
        self.assertIn('for name in ("uint16", "uint32", "uint64"):', utils)
        self.assertIn("getattr(torch, name, None)", utils)
        self.assertIn("getattr(infinicore, name)", utils)

    def test_from_numpy_converts_before_taking_an_owning_copy(self) -> None:
        source = read_source("python/infinicore/tensor.py")
        binding = read_source("csrc/infinicore/src/pybind11/tensor.hpp")

        self.assertIn("infinicore_to_numpy_dtype", source)
        self.assertIn("np.ascontiguousarray(array, dtype=numpy_dtype)", source)
        self.assertIn("_infinicore._from_numpy_copy(owner", source)
        self.assertNotIn("result.copy_(source)", source)

        self.assertIn('"_from_numpy_copy"', binding)
        self.assertIn("py::buffer", binding)
        self.assertIn("std::memcpy", binding)

    def test_to_numpy_is_owned_by_the_tensor_api(self) -> None:
        tensor = read_source("python/infinicore/tensor.py")
        generation = read_source("python/infinilm/generation/utils.py")

        self.assertIn("def to_numpy(self):", tensor)
        self.assertIn("infinicore_to_numpy_dtype", tensor)
        self.assertIn("ctypes.memmove", tensor)
        self.assertIn("return infini_tensor.to_numpy()", generation)
        self.assertNotIn("Tensor.to_numpy =", generation)

    def test_uninitialized_ones_factory_is_not_exported(self) -> None:
        tensor = read_source("python/infinicore/tensor.py")
        init = read_source("python/infinicore/__init__.py")
        binding = read_source("csrc/infinicore/src/pybind11/tensor.hpp")

        self.assertNotIn("def ones(", tensor)
        self.assertNotIn('"ones"', init)
        self.assertNotIn('m.def("ones"', binding)


if __name__ == "__main__":
    unittest.main()
