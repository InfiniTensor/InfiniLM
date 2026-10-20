# Contributing Guide

For development setup, see [Development Guide](#development-guide) below.

## Code

Please review these details before committing, especially for AI-generated code.

### General

1. Keep changes minimal — do not add what is not necessary.
2. Comments are not always better when abundant. Ideally, the code should be self-explanatory.
3. Files must end with a newline.
4. Use Markdown syntax (backtick-fenced) for identifiers in comments and error messages.
5. Comments and error messages must be in English.
6. Comments and error messages should follow the language's conventions first. If the language does not specify, use complete sentences — capitalize the first letter and end with punctuation.

### C++

1. Follow the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) strictly. Use the default `.clang-format`.
2. Operator parameters: inputs first, outputs last. For naming, prefer PyTorch conventions, then ONNX, then CUDA APIs. Attributes go between inputs and outputs.
3. Do not use exceptions. Use `assert` for error handling — debug builds will trigger assertion messages (which should include `__FILE__`, `__LINE__`, `__func__` at minimum), and release builds will have assertions compiled out.
4. Error and warning messages follow the [LLVM Coding Standards](https://llvm.org/docs/CodingStandards.html#error-and-warning-messages).
5. Kernel file naming (excluding extension): custom kernels without a well-known algorithm name should be named `kernel`. Multiple implementations use `kernel_v2`, `kernel_v3`, etc. Well-known algorithms use the algorithm name (e.g. `flash_attention_v2`). Library-based implementations use the library name (e.g. `blas`).
6. Separate kernel from kernel launcher. Launchers use `.h`. Kernels follow platform conventions (e.g. `.cuh`). Non-template kernels still require header/source separation (e.g. `.cuh` + `.cu`).
7. Initializer list order must match member declaration order.
8. One blank line between classes, between classes and functions, and between functions.
9. One blank line between each member (including both functions and variables) within a class.
10. One blank line before and after the contents of a namespace.

### Python

Follow [PEP 8](https://peps.python.org/pep-0008/) as the primary style guide. For anything PEP 8 does not cover in detail, refer to the [GDScript style guide](https://docs.godotengine.org/en/stable/tutorials/scripting/gdscript/gdscript_styleguide.html)—while it targets a different language, its non-syntax conventions are still applicable.

#### Additional Rules

1. **Comments** should be complete English sentences, starting with a capital letter and ending with punctuation. Use Markdown syntax when referencing code within comments.

2. **Error messages and framework conventions:** When a framework has an established convention (e.g., `pytest.skip` messages are typically lowercase without a trailing period), follow that convention. Otherwise, use the same rules as comments.

3. **Function signatures:** If a function has no docstring or comment, do not add a blank line between the function signature and the function body.

4. **Blank lines around control flow:** Add a blank line before and after `if`, `for`, and similar statements.

5. **Return statements:** Add a blank line before a `return` statement, unless it directly follows a control flow statement like `if` or `for`.

6. **Docstrings:** Follow [PEP 257](https://peps.python.org/pep-0257/) conventions.

## Commits

Commit messages must follow [Conventional Commits](https://www.conventionalcommits.org/).

## Pull Requests

1. Small PRs should be squashed. Large PRs may keep multiple commits, but each commit must be meaningful and well-formed.
2. PR titles follow the same Conventional Commits format as commit messages.
3. Pull requests should include smoke build and smoke test evidence for each affected platform. Full-suite or full-platform validation is still expected for high-risk changes, release preparation, maintainer spot checks, and changes that affect shared build, dispatch, wrapper generation, or cross-platform behavior.

## Branches

Branch names use the format `<type>/xxx-yyyy-zzzz`, where `<type>` matches the PR title's Conventional Commits type, and words are joined with hyphens.

---

# Development Guide

## Installation

Using Nvidia as an example:

```bash
pip install .[dev] \
  -C cmake.define.INFINI_RT_ROOT=/path/to/infini-rt-prefix \
  -C cmake.define.WITH_CPU=ON \
  -C cmake.define.WITH_NVIDIA=ON
```

`/path/to/infini-rt-prefix` is the InfiniRT install prefix, typically the value
used for InfiniRT's `CMAKE_INSTALL_PREFIX`. It should contain
`include/infini/rt.h` and `lib/libinfinirt.so`.

Auto-detection is supported for some platforms, so you can also let InfiniOps
detect the device backends while still pointing it at the installed InfiniRT:

```bash
pip install .[dev] -C cmake.define.INFINI_RT_ROOT=/path/to/infini-rt-prefix
```

> `[dev]` installs optional development dependencies (e.g. `pytest`) that are not needed for production but required for development and testing. After the first install, subsequent installs only need `pip install .`.

Platform maintainers can add auto-detection in `CMakeLists.txt` under the `if(AUTO_DETECT_DEVICES)` section.

## Testing

Run the full test suite when you need exhaustive validation:

```bash
python -m pytest
```

For routine development and pull requests, start with a smoke build plus the smoke test set on every affected platform:

```bash
python -m pip install .[dev] --no-build-isolation --no-deps \
  --config-settings=cmake.define.INFINI_RT_ROOT=/path/to/infini-rt-prefix \
  --config-settings=cmake.define.INFINI_OPS_SMOKE_BUILD=ON
python -m pytest tests -m smoke -q
```

The smoke set is a small representative subset that covers binding/codegen smoke tests, elementwise/layout operators, shape/list operators, matrix paths, and fused/model operators. It is intended to keep the default PR validation loop short while still exercising the main runtime surfaces. The default `python -m pytest` command remains the full suite.

The v1 timing target is NVIDIA smoke build plus smoke tests in under three minutes. `INFINI_OPS_SMOKE_BUILD=ON` narrows generated wrappers, bindings, and generated Torch ops to the smoke subset. Advanced users can override the subsets with `INFINI_OPS_OPS` for generated wrappers/bindings and `INFINI_OPS_TORCH_OPS` for generated Torch backend ops; both accept comma- or semicolon-separated operator names. Use full builds and full suites for high-risk changes, release prep, maintainer spot checks, or changes affecting shared build, dispatch, wrapper generation, or cross-platform behavior.

## Adding an Operator

Before defining an operator's public interface, follow
[Operator API Alignment](docs/operator-api-alignment.md) to select and document
its upstream API target.

1. **Base class** in `src/base/`: the class must inherit from `Operator<Op>` (e.g. `class Gemm : public Operator<Gemm>`). See `src/base/gemm.h`.
2. **Platform implementation** in `src/native/<category>/<platform>/ops/<op>/` (or `src/torch/ops/<op>/` for the PyTorch backend): the class must inherit from the base (e.g. `class Blas : public Gemm`). See `src/native/cuda/ops/gemm/blas.h` and `src/native/cuda/nvidia/ops/gemm/cublas.h`.
3. **Tests** in `tests/`:
   - Use `pytest.mark.parametrize` for parameterization. Dependent parameters go in one decorator (e.g. `@pytest.mark.parametrize("dtype, rtol, atol", ...)`); independent parameters use separate decorators, ordered by parameter declaration.
   - `dtype` and `device` parameterization is included by default. Override with explicit `pytest.mark.parametrize` if needed.
   - Use `pytest.mark.auto_act_and_assert` for automatic execution and comparison — just return a `Payload`. Requires that `func` and `ref` share the same calling convention and that all checked values are return values.
   - See `tests/test_add.py` and `tests/test_gemm.py`.

## Some Code Explanations

### `TypeMap` and `DataTypeMap`

Since `DataType` is an enum used to represent data types generically, we often need to map between `DataType` and native C++ types (e.g. `float`, `int32_t`).

- **`TypeMap`**: maps `DataType` to native types. Use the alias `TypeMapType` to get the type directly, e.g. `TypeMapType<dev, DataType::kFloat32>` is `float`. Note, the first template argument is a `Device::Type` since data types like float16 and bfloat16 are not the same across the platforms. Thus, a `Device::Type` is required to specify which native type a `DataType` maps to. 
- **`DataTypeMap`**: maps native types back to `DataType`. Use the alias `DataTypeMapValue`, e.g. `DataTypeMapValue<float>` is `DataType::kFloat32`.

### `DispatchFunc`

`DispatchFunc` is the runtime dispatch interface defined in `dispatcher.h`. It supports arbitrary types, multi-dispatch, and mixed-type dispatch with any return type.

#### Basic Usage

```cpp
DispatchFunc</* supported types */>(
    /* runtime value to dispatch on */,
    /* lambda with dispatched logic */,
    /* context string for error messages (recommended) */,
    /* forwarded args for the lambda (optional) */
);
```

#### Single-Type Dispatch (`Device::Type`)

```cpp
DispatchFunc<Device::Type::kCpu, Device::Type::kNvidia>(
    Device::Type::kNvidia,
    [](auto tag) {
      constexpr Device::Type Dev = decltype(tag)::value;
    },
    "DeviceTest");
```

#### Single-Type Dispatch (`DataType`)

```cpp
DataType dtype = DataType::kFloat32;
DispatchFunc<Device::Type::Cpu, FloatTypes>(
    dtype,
    [](auto tag) {
      using T = typename decltype(tag)::type;
      // Use T as the resolved native type.
    },
    "DataType Dispatch");
```

Dispatching `DataType` is a little bit special. 

1. Due to the previously mentioned `TypeMap` reason, a `Device::Type` is needed as the first template argument;

2. Since `DataType` is frequently used, the supported type list can use predefined shorthands from `data_type.h` (e.g. `FloatTypes` = `List<DataType::kFloat32, DataType::kFloat64>`). To combine shorthands, use `ConcatType` from `common/traits.h`:

```cpp
DispatchFunc<ConcatType<List<DataType::kFloat16>, FloatTypes>>(...);
```

#### Single-Type Dispatch (Custom Types)

For types other than `DataType` and `Device::Type`, pass the type as the first template argument:

```cpp
DispatchFunc<QuantMode, QuantMode::kNone, QuantMode::kWeightOnly>(
    QuantMode::kWeightOnly,
    [](auto tag) {
      constexpr QuantMode M = decltype(tag)::value;
    },
    "QuantDispatch");
```

This also works for native types like `int` (e.g. block sizes):

```cpp
DispatchFunc<int, 128, 256, 512, 1024>(
    runtime_block_size,
    [](auto tag) {
      constexpr int BlockSize = decltype(tag)::value;
    },
    "BlockSizeDispatch");
```

#### Multi-Dispatch (Same Type)

Use `List` boundaries to separate supported sets for each dispatched value. Pass runtime values in an initializer list:

```cpp
DispatchFunc<List<Device::Type::kCpu, Device::Type::kNvidia>,
             List<Device::Type::kAscend, Device::Type::kMetax>>(
      {Device::Type::kNvidia, Device::Type::kMetax},
      [](auto tag1, auto tag2) {
        constexpr Device::Type D1 = decltype(tag1)::value;
        constexpr Device::Type D2 = decltype(tag2)::value;
      },
      "MultiDeviceTest");
```

Similarly, `DataType` requires a `Device::Type` at the front: 

```cpp
DispatchFunc<Device::Type::kCpu, FloatTypes, List<DataType::kInt32, DataType::kInt64>>(
    {DataType::kFloat64, DataType::kInt32},
    [](auto tag1, auto tag2) {
      using T1 = typename decltype(tag1)::type;
      using T2 = typename decltype(tag2)::type;
    },
    "MultiDataTypeTest");
```

#### Mixed Multi-Type Dispatch

When dispatching different types simultaneously (e.g. `DataType` + `Device::Type`, or `DataType` + block size), cast values to `int64_t` and use `ListGet<N>` from `common/traits.h`:

```cpp
DispatchFunc<FloatTypes, List<Device::Type::kCpu, Device::Type::kNvidia>>(
    {static_cast<int64_t>(DataType::kFloat32),
     static_cast<int64_t>(Device::Type::kNvidia)},
    [](auto list_tag) {
      constexpr DataType DT = static_cast<DataType>(ListGet<0>(list_tag));
      constexpr Device::Type Dev = static_cast<Device::Type>(ListGet<1>(list_tag));
      using T = TypeMapType<Device::Type::kCpu, DT>;
    },
    "MixedDispatch");
```

Note that in mixed multi-type dispatch, `DataType` is not treated specially. Therefore, we neither should nor can place `Device::Type` at the front of the `DataType` list. Inside the lambda, we obtain it as a `DataType` and then convert it to the native type if needed. 

If `DT` is not used within the lambda, you can inline its definition directly into the `using T = ...` statement, like this: 

```cpp
using T = TypeMapType<Device::Type::kCpu, ListGet<0>(list_tag)>;
```

## Troubleshooting

1. **`no such option: -C` during install**: Upgrade pip with `python -m pip install --upgrade pip`.
2. **Segmentation fault during tests**: Run `pytest -n 1`.
3. **`Unknown CMake command "pybind11_add_module"`**: Install pybind11 with `pip install pybind11[global]`. See the [pybind11 installation guide](https://pybind11.readthedocs.io/en/stable/installing.html).
4. **Auto-detection (`AUTO_DETECT_DEVICES`) fails**: Some machines may not expose devices in expected paths (e.g. `/dev/nvidia*`). Use explicit CMake defines instead (e.g. `-C cmake.define.WITH_NVIDIA=ON`).
5. **InfiniRT headers or library are not found**: Set `INFINI_RT_ROOT` to the InfiniRT install prefix, or set `INFINI_RT_INCLUDE_DIRS` and `INFINI_RT_LIBRARY` explicitly.
6. **`ImportError: libinfinirt.so: cannot open shared object file`**: Reinstall InfiniOps from a build that was configured with the InfiniRT install prefix so the Python wheel can bundle the runtime library.
7. **`bash: pytest: command not found`**: Use `python -m pytest`.
8. **`CUBLAS_STATUS_INVALID_VALUE` in `cublasSgemmStridedBatched`**: PyTorch version issue. Downgrade to `torch<=2.9.1`.
