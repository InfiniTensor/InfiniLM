import importlib.util
import pathlib
import re
import sys

import pytest


def _load_generator_module():
    path = (
        pathlib.Path(__file__).resolve().parents[1] / "scripts" / "generate_wrappers.py"
    )
    spec = importlib.util.spec_from_file_location("generate_wrappers_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


def test_generated_dispatch_keeps_scalar_and_optional_tensor_overloads_distinct(
    monkeypatch, tmp_path
):
    module = _load_generator_module()
    base_header = tmp_path / "clamp.h"
    base_header.write_text(
        """
class Clamp {
 public:
  virtual void operator()(const Tensor input, const int64_t min,
                          const int64_t max, Tensor out) const = 0;
  virtual void operator()(const Tensor input, const std::optional<Tensor> min,
                          const std::optional<Tensor> max, Tensor out) const = 0;
};
"""
    )
    monkeypatch.setattr(module, "_find_base_header", lambda op_name: base_header)

    operator = module._Operator(
        "clamp",
        constructors=[
            module._ParsedFunction(
                [
                    module._ParsedArgument("const Tensor", "input"),
                    module._ParsedArgument("const int64_t", "min"),
                    module._ParsedArgument("const int64_t", "max"),
                    module._ParsedArgument("Tensor", "out"),
                ]
            ),
            module._ParsedFunction(
                [
                    module._ParsedArgument("const Tensor", "input"),
                    module._ParsedArgument("const std::optional<Tensor>", "min"),
                    module._ParsedArgument("const std::optional<Tensor>", "max"),
                    module._ParsedArgument("Tensor", "out"),
                ]
            ),
        ],
        calls=[],
    )

    declarations, _ = module._generate_generated_dispatch_entries(operator)

    text = "\n".join(declarations)
    binding = module._generate_pybind11(operator)

    assert (
        "py::init([](py::object input, const int64_t min, const int64_t max, py::object out)"
    ) in binding

    assert (
        "MakeClamp(const Config& config, Tensor input, "
        "const int64_t min, const int64_t max, "
        "Tensor out)"
    ) in text
    assert (
        "MakeClamp(const Config& config, Tensor input, "
        "std::optional<Tensor> min, std::optional<Tensor> max, Tensor out)"
    ) in text


def test_operator_call_instantiations_externalize_default_implementation_lookup():
    module = _load_generator_module()
    operator = module._Operator(
        "abs",
        constructors=[],
        calls=[
            module._ParsedFunction(
                [
                    module._ParsedArgument("const Tensor", "input"),
                    module._ParsedArgument("Tensor", "out"),
                ]
            )
        ],
    )

    declarations, definitions = module._generate_operator_call_instantiation_entries(
        operator
    )

    signature = (
        "std::size_t "
        "Operator<::infini::ops::Abs>::DefaultImplementationIndex(Device::Type);"
    )
    assert f"extern template {signature}" in declarations
    assert f"template {signature}" in definitions

    default_config_signature = (
        "Config Operator<::infini::ops::Abs>::DefaultConfig(Device::Type);"
    )
    assert declarations.count(f"extern template {default_config_signature}") == 1
    assert definitions.count(f"template {default_config_signature}") == 1
    eligibility = (
        "bool Operator<::infini::ops::Abs>::CanMake<Tensor>"
        "(const Config& config, const Tensor input, const Tensor& out);"
    )
    assert f"extern template {eligibility}" in declarations
    assert f"template {eligibility}" in definitions


def test_operator_call_instantiations_externalize_active_implementation_query():
    module = _load_generator_module()
    operator = module._Operator("add", constructors=[], calls=[])

    declarations, definitions = module._generate_operator_call_instantiation_entries(
        operator
    )

    signature = (
        "std::vector<std::size_t> "
        "Operator<::infini::ops::Add>::active_implementation_indices(Device::Type);"
    )
    assert f"extern template {signature}" in declarations
    assert f"template {signature}" in definitions


def test_public_factory_normalizes_formal_scalar_and_tensor_list_arguments():
    module = _load_generator_module()
    signature = module._ParsedFunction(
        [
            module._ParsedArgument("const std::vector<Tensor>", "inputs"),
            module._ParsedArgument("const double", "alpha"),
            module._ParsedArgument("Tensor", "out"),
        ]
    )
    operator = module._Operator("cat", constructors=[signature], calls=[signature])
    declarations, definitions = module._generate_operator_call_instantiation_entries(
        operator
    )
    header = "\n".join(declarations)
    source = "\n".join(definitions)
    assert "struct detail::OperatorFactory<::infini::ops::Cat>" in header
    assert (
        "Make(const Config& config, const std::vector<Tensor>& inputs, const double& alpha, const Tensor& out)"
        in header
    )
    assert "Operator<::infini::ops::Cat>::Make(config, inputs, alpha, out)" in source
    assert (
        "Operator<::infini::ops::Cat>::Call(handle, config, inputs, alpha, out)"
        in source
    )


def test_operator_call_instantiations_keep_scalar_and_optional_tensor_overloads_distinct(
    monkeypatch, tmp_path
):
    module = _load_generator_module()
    base_header = tmp_path / "clamp.h"
    base_header.write_text(
        """
class Clamp {
 public:
  virtual void operator()(const Tensor input, const int64_t min,
                          const int64_t max, Tensor out) const = 0;
  virtual void operator()(const Tensor input, const std::optional<Tensor> min,
                          const std::optional<Tensor> max, Tensor out) const = 0;
};
"""
    )
    monkeypatch.setattr(module, "_find_base_header", lambda op_name: base_header)

    operator = module._Operator(
        "clamp",
        constructors=[],
        calls=[
            module._ParsedFunction(
                [
                    module._ParsedArgument("const Tensor", "input"),
                    module._ParsedArgument("const int64_t", "min"),
                    module._ParsedArgument("const int64_t", "max"),
                    module._ParsedArgument("Tensor", "out"),
                ]
            ),
            module._ParsedFunction(
                [
                    module._ParsedArgument("const Tensor", "input"),
                    module._ParsedArgument("const std::optional<Tensor>", "min"),
                    module._ParsedArgument("const std::optional<Tensor>", "max"),
                    module._ParsedArgument("Tensor", "out"),
                ]
            ),
        ],
    )

    declarations, _ = module._generate_operator_call_instantiation_entries(operator)

    text = "\n".join(declarations)

    assert "Call<Tensor, int64_t, int64_t, Tensor>" in text
    assert (
        "Call<Tensor, std::optional<Tensor>, std::optional<Tensor>, Tensor>"
    ) in text


def test_extractor_prefers_header_types_for_reused_parameter_names(
    monkeypatch, tmp_path
):
    module = _load_generator_module()
    base_header = tmp_path / "histogram.h"
    base_header.write_text(
        """
class [[deprecated("Use HistogramV2 instead.")]] Histogram {
 public:
  virtual void operator()(const Tensor input, const Tensor bins,
                          const std::optional<Tensor> weight,
                          const bool density, Tensor hist,
                          Tensor bin_edges) const = 0;
  virtual void operator()(const Tensor input, const int64_t bins,
                          const std::optional<std::vector<double>> range,
                          const std::optional<Tensor> weight,
                          const bool density, Tensor hist,
                          Tensor bin_edges) const = 0;
};
"""
    )
    monkeypatch.setattr(module, "_find_base_header", lambda op_name: base_header)

    clang_calls = [
        module._ParsedFunction(
            [
                module._ParsedArgument("const int", "input"),
                module._ParsedArgument("const int", "bins"),
                module._ParsedArgument("const int", "weight"),
                module._ParsedArgument("const bool", "density"),
                module._ParsedArgument("int", "hist"),
                module._ParsedArgument("int", "bin_edges"),
            ]
        ),
        module._ParsedFunction(
            [
                module._ParsedArgument("const int", "input"),
                module._ParsedArgument("const int64_t", "bins"),
                module._ParsedArgument("const int", "range"),
                module._ParsedArgument("const int", "weight"),
                module._ParsedArgument("const bool", "density"),
                module._ParsedArgument("int", "hist"),
                module._ParsedArgument("int", "bin_edges"),
            ]
        ),
    ]

    operator = module._Operator(
        "histogram",
        constructors=[],
        calls=module._prefer_header_type_spellings(
            clang_calls, module._parse_operator_header("histogram").calls
        ),
    )

    declarations, _ = module._generate_operator_call_instantiation_entries(operator)
    text = "\n".join(declarations)

    assert (
        "Call<Tensor, int64_t, std::optional<std::vector<double>>, "
        "std::optional<Tensor>, bool, Tensor, Tensor>"
    ) in text
    assert "Call<Tensor, Tensor, int, std::optional<Tensor>" not in text


def test_vector_classification_is_overload_local_and_preserves_nesting(
    monkeypatch, tmp_path
):
    module = _load_generator_module()
    base_header = tmp_path / "max_unpool3d.h"
    base_header.write_text(
        """
class MaxUnpool3d {
 public:
  void operator()(const Tensor input, const Tensor indices,
                  const std::vector<int64_t> kernel_size,
                  const std::optional<std::vector<int64_t>> stride,
                  const std::vector<int64_t> padding,
                  const std::optional<std::vector<int64_t>> output_size,
                  Tensor out) const;
  virtual void operator()(const Tensor input, const Tensor indices,
                          const std::vector<int64_t> output_size,
                          const std::vector<int64_t> stride,
                          const std::vector<int64_t> padding,
                          Tensor out) const = 0;
};
"""
    )
    monkeypatch.setattr(module, "_find_base_header", lambda op_name: base_header)

    canonical = module._ParsedFunction(
        [
            module._ParsedArgument("const Tensor", "input"),
            module._ParsedArgument("const Tensor", "indices"),
            module._ParsedArgument("const std::vector<int64_t>", "kernel_size"),
            module._ParsedArgument(
                "const std::optional<std::vector<int64_t>>", "stride"
            ),
            module._ParsedArgument("const std::vector<int64_t>", "padding"),
            module._ParsedArgument(
                "const std::optional<std::vector<int64_t>>", "output_size"
            ),
            module._ParsedArgument("Tensor", "out"),
        ]
    )
    legacy = module._ParsedFunction(
        [
            module._ParsedArgument("const Tensor", "input"),
            module._ParsedArgument("const Tensor", "indices"),
            module._ParsedArgument("const std::vector<int64_t>", "output_size"),
            module._ParsedArgument("const std::vector<int64_t>", "stride"),
            module._ParsedArgument("const std::vector<int64_t>", "padding"),
            module._ParsedArgument("Tensor", "out"),
        ]
    )
    nested = module._ParsedFunction(
        [
            module._ParsedArgument("const Tensor", "a"),
            module._ParsedArgument("const Tensor", "b"),
            module._ParsedArgument("const std::vector<std::vector<int64_t>>", "dims"),
            module._ParsedArgument("Tensor", "out"),
        ]
    )
    operator = module._Operator(
        "max_unpool3d", constructors=[], calls=[canonical, legacy, nested]
    )

    binding = module._generate_pybind11(operator)
    dispatch, _ = module._generate_generated_dispatch_entries(operator)
    instantiations, _ = module._generate_operator_call_instantiation_entries(operator)
    generated = "\n".join((binding, *dispatch, *instantiations))

    assert "const std::optional<std::vector<int64_t>> stride" in generated
    assert "const std::optional<std::vector<int64_t>> output_size" in generated
    assert "const std::vector<int64_t> stride" in generated
    assert "const std::vector<int64_t> output_size" in generated
    assert "const std::vector<std::vector<int64_t>> dims" in generated


def test_vector_fallback_does_not_override_string_overload(monkeypatch, tmp_path):
    module = _load_generator_module()
    base_header = tmp_path / "conv1d.h"
    base_header.write_text(
        "class Conv1d { public: "
        "Conv1d(const Tensor input, const std::vector<int64_t> padding, Tensor out); "
        "Conv1d(const Tensor input, const std::string padding, Tensor out); };"
    )
    monkeypatch.setattr(module, "_find_base_header", lambda op_name: base_header)

    vector = module._ParsedArgument("const int", "padding")
    string = module._ParsedArgument("const std::string", "padding")
    vector_params = module._find_vector_int64_params("conv1d")

    assert module._vector_int64_kind(vector, vector_params, set()) == "vector"
    assert module._vector_int64_kind(string, vector_params, set()) is None


def test_pybind_default_implementation_uses_first_active_index(monkeypatch, tmp_path):
    module = _load_generator_module()
    base_header = tmp_path / "mul.h"
    base_header.write_text(
        """
class Mul {
 public:
  Mul(const Tensor input, const Tensor other, Tensor out);
  virtual void operator()(const Tensor input, const Tensor other, Tensor out) const = 0;
};
"""
    )
    monkeypatch.setattr(module, "_find_base_header", lambda op_name: base_header)

    operator = module._Operator(
        "mul",
        constructors=[
            module._ParsedFunction(
                [
                    module._ParsedArgument("const Tensor", "input"),
                    module._ParsedArgument("const Tensor", "other"),
                    module._ParsedArgument("Tensor", "out"),
                ]
            )
        ],
        calls=[
            module._ParsedFunction(
                [
                    module._ParsedArgument("const Tensor", "input"),
                    module._ParsedArgument("const Tensor", "other"),
                    module._ParsedArgument("Tensor", "out"),
                ]
            )
        ],
    )

    text = module._generate_pybind11(operator)

    assert '#include "tuning.h"' in text
    assert "std::size_t DefaultImplementationIndexForMul" in text
    assert (
        "return ::infini::ops::Mul::DefaultConfig(dev_type).implementation_index();"
    ) in text
    assert "detail::AutomaticDefault<::infini::ops::Mul>::value" not in text
    assert "return Config::kAutomaticImplementation;" not in text
    assert "return indices.front();" not in text
    assert (
        "config.set_implementation_index("
        "DefaultImplementationIndexForMul(DeviceFromPybind11Handle(input).type()))"
    ) in text
    assert "std::optional<std::size_t> implementation_index" in text
    assert (
        "if (implementation_index.has_value()) {\n"
        "      config.set_implementation_index(*implementation_index);\n"
        "    } else if (!TuningManager::Instance().IsEnabled()) {\n"
        "      config.set_implementation_index(\n"
        "          DefaultImplementationIndexForMul("
        "converted_first_tensor.device().type()));\n"
        "    }"
    ) in text
    assert "auto converted_first_tensor{TensorFromPybind11Handle(input)};" in text
    assert "std::move(converted_first_tensor)" not in text
    assert text.count("DeviceFromPybind11Handle(input)") == 1
    assert "implementation_index.value_or(" not in text
    assert 'py::arg("implementation_index") = py::none()' in text


def test_pybind_reuses_first_vector_tensor_conversion(monkeypatch, tmp_path):
    module = _load_generator_module()
    base_header = tmp_path / "cat.h"
    base_header.write_text(
        """
class Cat {
 public:
  virtual void operator()(const std::vector<Tensor> inputs, Tensor out) const = 0;
};
"""
    )
    monkeypatch.setattr(module, "_find_base_header", lambda op_name: base_header)

    operator = module._Operator(
        "cat",
        constructors=[],
        calls=[
            module._ParsedFunction(
                [
                    module._ParsedArgument("const std::vector<Tensor>", "inputs"),
                    module._ParsedArgument("Tensor", "out"),
                ]
            )
        ],
    )

    text = module._generate_pybind11(operator)

    assert (
        "auto converted_first_tensor{VectorTensorFromPybind11Handle(inputs)};" in text
    )
    assert "generated_dispatch::CallCat(handle, config, converted_first_tensor," in text
    assert "converted_first_tensor.at(0).device().type()" in text
    assert "std::move(converted_first_tensor)" not in text
    assert "DeviceFromPybind11Handle(inputs.at(0))" not in text
    declarations, _ = module._generate_operator_call_instantiation_entries(operator)
    assert any(
        "::CanMake<Tensor>(const Config& config, "
        "const std::vector<Tensor> inputs, const Tensor& out)" in declaration
        for declaration in declarations
    )


def test_optional_tensor_vector_is_preserved_across_generated_wrappers(
    monkeypatch, tmp_path
):
    module = _load_generator_module()
    base_header = tmp_path / "index.h"
    base_header.write_text(
        """
class Index {
 public:
  Index(const Tensor input,
        const std::vector<std::optional<Tensor>> indices, Tensor out) {}

  virtual void operator()(
      const Tensor input,
      const std::vector<std::optional<Tensor>> indices, Tensor out) const = 0;
};
"""
    )
    monkeypatch.setattr(module, "_find_base_header", lambda op_name: base_header)

    operator = module._parse_operator_header("index")
    binding = module._generate_pybind11(operator)
    dispatch_declarations, _ = module._generate_generated_dispatch_entries(operator)
    instantiation_declarations, _ = (
        module._generate_operator_call_instantiation_entries(operator)
    )

    assert "std::vector<py::object> indices" in binding
    assert "VectorOptionalTensorFromPybind11Handle(indices)" in binding
    assert "std::optional<py::object> indices" not in binding
    assert 'py::arg("indices") = py::none()' not in binding

    dispatch_text = "\n".join(dispatch_declarations)
    assert "std::vector<std::optional<Tensor>> indices" in dispatch_text

    instantiation_text = "\n".join(instantiation_declarations)
    assert (
        "Call<Tensor, std::vector<std::optional<Tensor>>, Tensor>" in instantiation_text
    )


_DTYPE_OP_SOURCE = """
namespace infini::ops {

struct Tensor {};

enum class DataType {};

template <typename T>
class Operator {};

class DtypeOp : public Operator<DtypeOp> {
 public:
  DtypeOp(const Tensor input, const DataType out_dtype, Tensor out) {}

  virtual void operator()(const Tensor input, const DataType out_dtype,
                          Tensor out) const = 0;
};

}  // namespace infini::ops
"""


def _make_profile_operator(module, tmp_path, monkeypatch):
    base_header = tmp_path / "profile_op.h"
    base_header.write_text(
        """
class ProfileOp {
 public:
  ProfileOp(const Tensor input, Tensor out);
  virtual void operator()(const Tensor input, Tensor out) const = 0;
  virtual void operator()(const Tensor input, const double alpha,
                          Tensor out) const = 0;
};
"""
    )
    monkeypatch.setattr(module, "_find_base_header", lambda op_name: base_header)

    arguments = [
        module._ParsedArgument("const Tensor", "input"),
        module._ParsedArgument("Tensor", "out"),
    ]
    scaled_arguments = [
        module._ParsedArgument("const Tensor", "input"),
        module._ParsedArgument("const double", "alpha"),
        module._ParsedArgument("Tensor", "out"),
    ]

    return module._Operator(
        "profile_op",
        constructors=[module._ParsedFunction(arguments)],
        calls=[
            module._ParsedFunction(arguments),
            module._ParsedFunction(scaled_arguments),
        ],
    )


def test_generated_free_calls_start_with_binding_body_profile_scope(
    tmp_path, monkeypatch
):
    module = _load_generator_module()
    operator = _make_profile_operator(module, tmp_path, monkeypatch)

    text = module._generate_pybind11(operator)

    scope = (
        "[[maybe_unused]] HostRangeScope host_range_binding_body{\n"
        "        HostRangeLayer::kBindingBody};"
    )
    assert '#include "host_range_profiler.h"' in text
    free_call_bodies = re.findall(
        r'm\.def\("profile_op", \[\]\([^)]*\) \{\n(.*?)\n  \},',
        text,
        flags=re.DOTALL,
    )
    assert len(free_call_bodies) == 2

    for body in free_call_bodies:
        assert body.lstrip().startswith(scope)
        assert body.count(scope) == 1
        assert body.index(scope) < body.index("Handle handle;")
        assert body.index(scope) < body.index("Config config;")


def test_generated_ops_module_exposes_host_range_profile_controls_once():
    module = _load_generator_module()

    text = module._generate_ops_module_source(["BindProfileOp"])

    assert '#include "host_range_profiler.h"' in text
    expected_targets = {
        "_host_range_profile_compiled": "HostRangeProfiler::IsCompiled",
        "_host_range_profile_start": "HostRangeProfiler::Start",
        "_host_range_profile_stop": "HostRangeProfiler::Stop",
        "_host_range_profile_calibrate": "HostRangeProfiler::Calibrate",
    }

    for name, target in expected_targets.items():
        marker = f'm.def("{name}"'
        assert text.count(marker) == 1
        binding = text.split(marker, maxsplit=1)[1].split("\n  m.def(", maxsplit=1)[0]
        assert target in binding


@pytest.mark.parametrize("monolithic", [False, True])
def test_generated_ops_module_initializes_tuning(monolithic):
    module = _load_generator_module()

    text = module._generate_ops_module_source(
        ["BindAdd"],
        op_includes=['#include "base/add.h"'],
        monolithic=monolithic,
    )

    assert text.count('#include "tuning.h"') == 1
    assert text.count("TuningManager::Instance().InitializeFromEnvironment();") == 1
    assert text.count("BindHostRangeProfileControls(m);") == 1


def test_iluvatar_custom_compilers_receive_host_range_profile_definition():
    cmake = (pathlib.Path(__file__).parents[1] / "src" / "CMakeLists.txt").read_text(
        encoding="utf-8"
    )

    for definitions in (
        "_iluvatar_call_instantiation_defs",
        "_iluvatar_dispatch_defs",
    ):
        assert (
            f"list(APPEND {definitions}\n"
            "                -DINFINI_OPS_ENABLE_HOST_RANGE_PROFILING=1)"
        ) in cmake


def test_torch_system_compiler_receives_host_range_profile_definition():
    cmake = (pathlib.Path(__file__).parents[1] / "src" / "CMakeLists.txt").read_text(
        encoding="utf-8"
    )

    assert (
        "list(APPEND _torch_extra_flags\n"
        '                "-DINFINI_OPS_ENABLE_HOST_RANGE_PROFILING=1")'
    ) in cmake


def test_explicit_op_allowlist_precedes_implicit_ops_json():
    cmake = (pathlib.Path(__file__).parents[1] / "src" / "CMakeLists.txt").read_text(
        encoding="utf-8"
    )
    wrapper = (
        pathlib.Path(__file__).parents[1] / "scripts" / "generate_wrappers.py"
    ).read_text(encoding="utf-8")

    assert (
        'elseif(NOT INFINI_OPS_OPS AND EXISTS "${PROJECT_SOURCE_DIR}/ops.json")'
        in cmake
    )
    assert 'if(INFINI_OPS_OPS AND NOT INFINI_OPS_OPS MATCHES "\\\\.json$")' in cmake
    assert "implicit_config_path" not in wrapper


def test_operator_selection_tracks_implementation_header_changes():
    cmake = (pathlib.Path(__file__).parents[1] / "src" / "CMakeLists.txt").read_text(
        encoding="utf-8"
    )

    for root in ("native", "torch", "ninetoothed", "linked", "triton"):
        assert f'"${{PROJECT_SOURCE_DIR}}/src/{root}/*.h"' in cmake
    assert "${_infini_ops_implementation_headers}" in cmake
    assert '"${PROJECT_SOURCE_DIR}/scripts/generate_torch_ops.py"' in cmake
    assert '"${PROJECT_SOURCE_DIR}/scripts/torch_ops.yaml"' in cmake


def test_generated_dispatch_calls_start_with_dispatch_profile_scope(
    tmp_path, monkeypatch
):
    module = _load_generator_module()
    operator = _make_profile_operator(module, tmp_path, monkeypatch)
    _, definitions = module._generate_generated_dispatch_entries(operator)

    text = module._generate_generated_dispatch_source([], definitions)

    scope = (
        "[[maybe_unused]] HostRangeScope host_range_dispatch_call{\n"
        "      HostRangeLayer::kDispatchCall};"
    )
    assert '#include "host_range_profiler.h"' in text
    assert '#include "infini/operator_call_instantiations.h"' in text
    dispatch_call_bodies = re.findall(
        r"void CallProfileOp\([^)]*\) \{\n(.*?)\n\}",
        text,
        flags=re.DOTALL,
    )
    assert len(dispatch_call_bodies) == 2

    for body in dispatch_call_bodies:
        assert body.lstrip().startswith(scope)
        assert body.count(scope) == 1
        assert body.index(scope) < body.index(
            "Operator<::infini::ops::ProfileOp>::Call"
        )


def test_generated_cache_clear_has_a_backend_independent_core_source(
    tmp_path, monkeypatch
):
    module = _load_generator_module()
    operator = _make_profile_operator(module, tmp_path, monkeypatch)

    _, definitions = module._generate_generated_dispatch_entries(operator)
    dispatch_text = "\n".join(definitions)
    core_text = module._generate_cache_clear_dispatch_source(["profile_op"])

    assert (
        "#if !defined(INFINI_OPS_USE_OPERATOR_CALL_INSTANTIATIONS)\n"
        "void ClearCacheForProfileOp()"
    ) in dispatch_text
    assert "INFINI_OPS_BUILD_CORE_DISPATCH" not in dispatch_text
    assert core_text.count("void ClearCacheForProfileOp()") == 1
    assert '#include "base/profile_op.h"' in core_text
    assert "Operator<::infini::ops::ProfileOp>::clear_cache();" in core_text


def test_pybind_converts_data_type_arguments_from_torch_dtype(tmp_path, monkeypatch):
    text = _generate_binding("dtype_op", tmp_path, monkeypatch, _DTYPE_OP_SOURCE)

    assert "py::object out_dtype" in text
    assert "DataTypeFromPybind11Handle(out_dtype)" in text
    assert 'py::arg("out_dtype")' in text


def test_legacy_c_converts_data_type_arguments_from_infini_dtype(tmp_path, monkeypatch):
    module = _load_generator_module()
    base_header = tmp_path / "dtype_op.h"
    base_header.write_text(_DTYPE_OP_SOURCE)
    monkeypatch.setattr(module, "_find_base_header", lambda op_name: base_header)
    operator = module._OperatorExtractor()("dtype_op")

    source, header = module._generate_legacy_c(operator, ())

    assert "const infiniDtype_t out_dtype" in header
    assert "DataTypeFromInfiniDType(out_dtype)" in source


def test_normalize_op_allowlist_accepts_spaces_and_commas():
    module = _load_generator_module()

    assert module._normalize_op_allowlist(["add,mul", " cast ", "", "gemm"]) == [
        "add",
        "mul",
        "cast",
        "gemm",
    ]


def test_filter_ops_preserves_allowlist_order_and_skips_unavailable_ops():
    module = _load_generator_module()
    ops = {"add": ["add_impl"], "mul": ["mul_impl"], "gemm": ["gemm_impl"]}

    assert module._filter_ops(ops, ["gemm", "add"]) == {
        "gemm": ["gemm_impl"],
        "add": ["add_impl"],
    }
    assert module._filter_ops(ops, ["add", "missing"]) == {"add": ["add_impl"]}


def test_filter_ops_strict_rejects_unavailable_ops():
    module = _load_generator_module()
    ops = {"add": ["add_impl"]}

    try:
        module._filter_ops(ops, ["add", "missing"], strict=True)
    except ValueError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("strict unknown ops should fail")


def test_select_ops_from_config_filters_implementation_slots(tmp_path):
    module = _load_generator_module()
    slot_0 = tmp_path / "slot_0.h"
    slot_16 = tmp_path / "slot_16.h"
    slot_0.write_text("class Operator<Add, Device::Type::kCpu> : public Add {};")
    slot_16.write_text("class Operator<Add, Device::Type::kCpu, 16> : public Add {};")
    implementation_0 = module._Implementation(slot_0, "native")
    implementation_16 = module._Implementation(slot_16, "native")
    config = {
        "add": {"headers": None, "implementations": (16,)},
    }

    assert module._select_ops_from_config(
        {"add": [implementation_0, implementation_16]}, config, "ops.json"
    ) == {"add": [implementation_16]}


def test_select_ops_from_config_rejects_missing_slot(tmp_path):
    module = _load_generator_module()
    slot_0 = tmp_path / "slot_0.h"
    slot_0.write_text("class Operator<Add, Device::Type::kCpu> : public Add {};")
    implementation_0 = module._Implementation(slot_0, "native")
    config = {
        "add": {"headers": None, "implementations": (16,)},
    }

    try:
        module._select_ops_from_config({"add": [implementation_0]}, config, "ops.json")
    except ValueError as exc:
        assert "slot(s) 16" in str(exc)
    else:
        raise AssertionError("unavailable implementation slot should fail")


def test_select_ops_from_config_preserves_path_and_backend_descriptors():
    module = _load_generator_module()
    config = {
        "add": {
            "headers": [
                "src/native/cpu/ops/add/add.h",
                {"path": "custom/add.h", "backend": "custom"},
            ],
            "implementations": None,
        },
    }

    assert module._select_ops_from_config(
        {
            "add": [
                module._Implementation(
                    pathlib.Path("src/native/cpu/ops/add/add.h"), "native"
                )
            ]
        },
        config,
        "ops.json",
    ) == {
        "add": [
            module._Implementation(
                pathlib.Path("src/native/cpu/ops/add/add.h"), "native"
            ),
            module._Implementation(pathlib.Path("custom/add.h"), "custom"),
        ]
    }


def test_linked_implementations_require_explicit_scan_flag(monkeypatch, tmp_path):
    module = _load_generator_module()
    src_dir = tmp_path / "moore" / "src"
    base_dir = src_dir / "base"
    provider_dir = src_dir / "linked" / "torch" / "metax" / "ops" / "silu_and_mul"
    vllm_header = provider_dir / "vllm.h"
    apex_header = provider_dir / "apex.h"
    base_dir.mkdir(parents=True)
    provider_dir.mkdir(parents=True)
    (base_dir / "silu_and_mul.h").write_text("class SiluAndMul {};\n")
    vllm_header.write_text("class Operator<SiluAndMul, Device::Type::kMetax, 16> {};\n")
    apex_header.write_text("class Operator<SiluAndMul, Device::Type::kMetax, 17> {};\n")
    monkeypatch.setattr(module, "_SRC_DIR", src_dir)
    monkeypatch.setattr(module, "_BASE_DIR", base_dir)
    monkeypatch.setattr(module, "_GENERATION_DIR", tmp_path / "generated")

    assert "silu_and_mul" not in module._get_all_ops(["metax"])
    assert "silu_and_mul" not in module._get_all_ops(["moore"], with_linked=True)
    linked_ops = module._get_all_ops(["metax"], with_linked=True)
    assert {implementation.path for implementation in linked_ops["silu_and_mul"]} == {
        vllm_header,
        apex_header,
    }


def test_write_text_if_changed_preserves_unchanged_mtime(tmp_path):
    module = _load_generator_module()
    path = tmp_path / "bindings.cc"
    path.write_text("same\n")
    before = path.stat().st_mtime_ns

    assert module._write_text_if_changed(path, "same\n") is False
    assert path.stat().st_mtime_ns == before

    assert module._write_text_if_changed(path, "different\n") is True
    assert path.read_text() == "different\n"
    assert path.stat().st_mtime_ns >= before


def test_remove_stale_files_keeps_expected_outputs(tmp_path):
    module = _load_generator_module()
    root = tmp_path / "generated"
    keep = root / "bindings" / "keep.cc"
    stale = root / "bindings" / "stale.cc"
    nested_stale = root / "src" / "foo" / "operator.cc"
    keep.parent.mkdir(parents=True)
    nested_stale.parent.mkdir(parents=True)
    keep.write_text("keep\n")
    stale.write_text("stale\n")
    nested_stale.write_text("stale\n")

    module._remove_stale_files(root, {keep})

    assert keep.exists()
    assert not stale.exists()
    assert not nested_stale.exists()
    assert not (root / "src" / "foo").exists()


def _generate_binding(op_name, tmp_path, monkeypatch, source):
    module = _load_generator_module()
    src_dir = tmp_path / "src"
    base_dir = src_dir / "base"
    base_dir.mkdir(parents=True)
    (base_dir / f"{op_name}.h").write_text(source)
    monkeypatch.setattr(module, "_SRC_DIR", src_dir)
    monkeypatch.setattr(module, "_BASE_DIR", base_dir)
    operator = module._OperatorExtractor()(op_name)

    return module._generate_pybind11(operator)


def test_mha_varlen_fwd_requires_out_binding(tmp_path, monkeypatch):
    text = _generate_binding(
        "mha_varlen_fwd",
        tmp_path,
        monkeypatch,
        """
#include <cstdint>
#include <optional>

namespace infini::ops {

struct Tensor {};

template <typename T>
class Operator {};

class MhaVarlenFwd : public Operator<MhaVarlenFwd> {
 public:
  MhaVarlenFwd(const Tensor q, const Tensor k, const Tensor v, Tensor out,
               const Tensor cu_seqlens_q, const Tensor cu_seqlens_k,
               std::optional<Tensor> block_table, float softmax_scale,
               bool is_causal, int64_t num_splits = 0) {}

  virtual void operator()(const Tensor q, const Tensor k, const Tensor v,
                          Tensor out, const Tensor cu_seqlens_q,
                          const Tensor cu_seqlens_k,
                          std::optional<Tensor> block_table,
                          float softmax_scale, bool is_causal,
                          int64_t num_splits = 0) const = 0;
};

}  // namespace infini::ops
""",
    )

    assert 'py::arg("out"), py::arg("cu_seqlens_q")' in text
    assert 'py::arg("num_splits") = 0' in text
    assert 'py::arg("out") = py::none()' not in text
    assert "std::optional<py::object> out" not in text
    assert "OptionalTensorFromPybind11Handle(out)" not in text


def test_mha_fwd_kvcache_requires_out_binding(tmp_path, monkeypatch):
    text = _generate_binding(
        "mha_fwd_kvcache",
        tmp_path,
        monkeypatch,
        """
#include <cstdint>
#include <optional>

namespace infini::ops {

struct Tensor {};

template <typename T>
class Operator {};

class MhaFwdKvcache : public Operator<MhaFwdKvcache> {
 public:
  MhaFwdKvcache(const Tensor q, const Tensor kcache, const Tensor vcache,
                std::optional<Tensor> k, std::optional<Tensor> v, Tensor out,
                float softmax_scale, bool is_causal,
                int64_t num_splits = 0) {}

  virtual void operator()(const Tensor q, const Tensor kcache,
                          const Tensor vcache, std::optional<Tensor> k,
                          std::optional<Tensor> v, Tensor out,
                          float softmax_scale, bool is_causal,
                          int64_t num_splits = 0) const = 0;
};

}  // namespace infini::ops
""",
    )

    assert 'py::arg("out"), py::arg("softmax_scale")' in text
    assert 'py::arg("num_splits") = 0' in text
    assert 'py::arg("out") = py::none()' not in text
    assert "std::optional<py::object> out" not in text
    assert "OptionalTensorFromPybind11Handle(out)" not in text


_TRITON_ADD_SOURCE = """


namespace infini::ops {

struct Tensor {};

class Add {
 public:
  Add(const Tensor input, const Tensor other, const double alpha, Tensor out) {}
  Add(const Tensor input, const Tensor other, Tensor out) {}

  virtual void operator()(const Tensor input, const Tensor other,
                          const double alpha, Tensor out) const = 0;
  virtual void operator()(const Tensor input, const Tensor other,
                          Tensor out) const = 0;
};

}  // namespace infini::ops
"""


def _make_backend_add(module, tmp_path, monkeypatch, backend):
    base_header = tmp_path / "base" / "add.h"
    base_header.parent.mkdir()
    base_header.write_text(_TRITON_ADD_SOURCE)
    monkeypatch.setattr(module, "_find_base_header", lambda op_name: base_header)

    parsed = module._OperatorExtractor()("add")
    implementation_path = tmp_path / backend / "ops" / "add" / "jit.h"
    implementation_path.parent.mkdir(parents=True)
    implementation_path.write_text("// JitConfig is intentionally irrelevant.\n")

    return module._Operator(
        parsed.name,
        parsed.constructors,
        parsed.calls,
        implementations=[
            module._Implementation(implementation_path, backend),
        ],
    )


def test_triton_binding_uses_backend_metadata_and_shared_config_parser(
    tmp_path, monkeypatch
):
    module = _load_generator_module()
    operator = _make_backend_add(module, tmp_path, monkeypatch, "triton")

    binding = module._generate_pybind11(operator)

    assert '#include "tuning.h"' in binding
    assert '#include "triton/jit/pybind11_config.h"' in binding
    assert "std::unique_ptr<Config> triton_config_ptr;" in binding
    assert "triton::jit::ConfigFromPyDict(*config_dict)" in binding
    assert "triton_config_ptr->set_implementation_index(" in binding
    assert "if (!config.needs_implementation_resolution())" in binding
    assert "config.implementation_index()" in binding
    assert "if (triton_config_ptr)" in binding
    assert "generated_dispatch::MakeAdd(*triton_config_ptr," in binding
    assert "return (*op)(handle," in binding
    assert "generated_dispatch::CallAdd(handle, config," in binding
    assert "std::move(converted_first_tensor)" not in binding
    assert "ValidateConfigFields" not in binding

    declarations, _ = module._generate_generated_dispatch_entries(operator)
    make_declarations = [
        declaration for declaration in declarations if " MakeAdd(" in declaration
    ]
    assert len(make_declarations) == len(operator.constructors)


def test_native_binding_does_not_scan_implementation_source_for_config_support(
    tmp_path, monkeypatch
):
    module = _load_generator_module()
    operator = _make_backend_add(module, tmp_path, monkeypatch, "native")

    binding = module._generate_pybind11(operator)

    assert "triton/jit/pybind11_config.h" not in binding
    assert "config_dict" not in binding
    assert "triton::jit::ConfigFromPyDict" not in binding


def test_implementation_index_records_structured_backend_metadata(tmp_path):
    module = _load_generator_module()
    implementation_path = tmp_path / "triton" / "ops" / "add" / "jit.h"
    implementation_path.parent.mkdir(parents=True)
    implementation_path.write_text("class Operator<Add, Device::Type::kNvidia>;\n")

    index = module._index_impl_headers([tmp_path], {"triton"})

    assert index["Add"] == [
        module._Implementation(implementation_path, "triton"),
    ]


def test_triton_config_is_only_exposed_for_constructor_shaped_calls():
    module = _load_generator_module()
    parsed = module._OperatorExtractor()("add_rms_norm")
    operator = module._Operator(
        parsed.name,
        parsed.constructors,
        parsed.calls,
        implementations=[
            module._Implementation(
                pathlib.Path("triton/ops/add_rms_norm/jit.h"),
                "triton",
            )
        ],
    )
    assert len(operator.constructors) == 1
    assert len(operator.calls) == 2

    binding = module._generate_pybind11(operator)

    declarations, _ = module._generate_generated_dispatch_entries(operator)
    make_declarations = [
        declaration for declaration in declarations if " MakeAddRmsNorm(" in declaration
    ]

    assert binding.count('py::arg("config") = py::none()') == 1
    assert binding.count("triton::jit::ConfigFromPyDict(*config_dict)") == 1
    assert len(make_declarations) == len(operator.constructors)


@pytest.mark.parametrize(
    "serialized, expected_path, expected_backend",
    (
        (
            "src/triton/ops/add/jit.h",
            pathlib.Path("src/triton/ops/add/jit.h"),
            "triton",
        ),
        (
            {"path": "custom/add.h", "backend": "custom"},
            pathlib.Path("custom/add.h"),
            "custom",
        ),
    ),
)
def test_implementation_json_accepts_legacy_and_structured_entries(
    serialized, expected_path, expected_backend
):
    module = _load_generator_module()

    implementation = module._implementation_from_json(serialized)

    assert implementation.path == expected_path
    assert implementation.backend == expected_backend


def test_triton_operator_discovery_uses_combined_operator_header(monkeypatch, tmp_path):
    module = _load_generator_module()
    src_dir = pathlib.Path(__file__).resolve().parents[1] / "src"
    monkeypatch.setattr(module, "_SRC_DIR", src_dir)
    monkeypatch.setattr(module, "_BASE_DIR", src_dir / "base")
    monkeypatch.setattr(module, "_GENERATION_DIR", tmp_path / "generated")

    ops = module._get_all_ops(["nvidia"], with_triton=True)

    assert {
        implementation.path.relative_to(src_dir).as_posix()
        for implementation in ops["add"]
        if implementation.backend == "triton"
    } == {"triton/ops/add/jit.h"}


def test_shared_triton_config_parser_has_one_explicit_schema():
    parser_header = (
        pathlib.Path(__file__).resolve().parents[1]
        / "src"
        / "triton"
        / "jit"
        / "pybind11_config.h"
    ).read_text()

    for field in (
        "auto_tuning",
        "num_warps",
        "num_stages",
        "constexprs",
        "warmup_milliseconds",
        "repetition_milliseconds",
        "keys",
        "candidates",
    ):
        assert f'"{field}"' in parser_header

    assert "namespace infini::ops::triton::jit" in parser_header
    assert "namespace detail" in parser_header


def test_cmake_configures_cambricon_half_compatibility():
    root = pathlib.Path(__file__).parents[1]
    source_cmake = (root / "src" / "CMakeLists.txt").read_text(encoding="utf-8")
    compatibility_header = (
        root / "src" / "native" / "cambricon" / "host_bang_compat.h"
    ).read_text(encoding="utf-8")

    assert "native/cambricon/host_bang_compat.h" in source_cmake
    assert "target_compile_options(infiniops PRIVATE" in source_cmake
    assert "target_compile_options(ops PRIVATE" in source_cmake
    assert source_cmake.count("_cambricon_host_bang_compat_option}") == 2
    assert "#include <bang_fp16.h>" in compatibility_header
    assert "#include <bang_bf16.h>" in compatibility_header
    assert "#define __mlu_host__ inline" in compatibility_header


def test_builtin_header_requires_enabled_provider():
    module = _load_generator_module()
    config = {
        "flash_attn_with_kvcache": {
            "headers": ["src/native/cuda/nvidia/ops/flash_attn_with_kvcache/kernel.h"],
            "implementations": None,
        }
    }
    with pytest.raises(ValueError, match="not enabled by this build"):
        module._select_ops_from_config({}, config, "ops.json")


def test_optional_provider_flags_with_absolute_source_root(monkeypatch):
    module = _load_generator_module()
    source = pathlib.Path(__file__).resolve().parents[1] / "src"
    monkeypatch.setattr(module, "_SRC_DIR", source)
    monkeypatch.setattr(module, "_BASE_DIR", source / "base")
    off = module._get_all_ops(["nvidia"])
    on = module._get_all_ops(["nvidia"], with_native_flash=True, with_native_w4=True)
    for name in (
        "flash_attn_with_kvcache",
        "flash_attn_varlen_func",
        "w4_marlin_gemm",
        "w4_marlin_repack",
        "w4_marlin_parameters",
    ):
        assert name not in off
        assert on[name]


def test_w4_public_api_selection_is_explicit(tmp_path):
    module = _load_generator_module()
    name = "w4_marlin_gemm"
    external = module._Implementation(tmp_path / "w4.h", "custom")
    config = {
        name: {
            "headers": [{"path": str(external.path), "backend": "custom"}],
            "implementations": None,
        }
    }
    assert module._select_ops_from_config({}, config, "ops.json") == {name: [external]}
    assert module._select_ops_from_config(
        {}, {name: {"public_api_only": True}}, "ops.json"
    ) == {name: []}
    with pytest.raises(ValueError, match="not available"):
        module._select_ops_from_config(
            {}, {name: {"headers": None, "implementations": (0,)}}, "ops.json"
        )
