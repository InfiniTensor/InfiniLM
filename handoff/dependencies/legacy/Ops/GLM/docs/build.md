# Build and Test

InfiniOps uses CMake, scikit-build-core, and Python packaging. The most common
entry is `python -m pip install` with CMake options passed through
`--config-settings`.

## Common CMake Options

| Option | Description | Default |
| --- | --- | :---: |
| `WITH_CPU` | Enable the CPU backend. | `OFF` |
| `WITH_NVIDIA` | Enable the NVIDIA CUDA backend. | `OFF` |
| `WITH_ILUVATAR` | Enable the Iluvatar CUDA-compatible backend. | `OFF` |
| `WITH_HYGON` | Enable the Hygon backend. | `OFF` |
| `WITH_THEAD` | Enable the T-Head CUDA-compatible backend. | `OFF` |
| `WITH_METAX` | Enable the MetaX backend. | `OFF` |
| `WITH_CAMBRICON` | Enable the Cambricon backend. | `OFF` |
| `WITH_MOORE` | Enable the Moore backend. | `OFF` |
| `WITH_ASCEND` | Enable the Ascend backend. | `OFF` |
| `WITH_TORCH` | Enable PyTorch C++ ATen-backed operators. | `OFF` |
| `WITH_LINKED` | Enable operators linked from installed third-party libraries. | `OFF` |
| `WITH_NINETOOTHED` | Enable NineToothed-generated kernels. | `OFF` |
| `WITH_TRITON` | Enable the NVIDIA Triton JIT backend. Requires Python bindings and CUDA Toolkit 12.0 or newer. | `OFF` |
| `AUTO_DETECT_DEVICES` | Auto-detect available device files. | `OFF` |
| `AUTO_DETECT_BACKENDS` | Auto-detect available backend packages. | `OFF` |
| `GENERATE_OPERATOR_CALL_INSTANTIATIONS` | Generate explicit C++ operator call instantiations. | `ON` |
| `GENERATE_PYTHON_BINDINGS` | Generate Python bindings. | `OFF` in raw CMake, `ON` in `pyproject.toml` |
| `INFINI_OPS_BUILD_DOCS` | Enable the Doxygen documentation target. | `OFF` |
| `INFINI_RT_ROOT` | InfiniRT install prefix containing `include/` and `lib/`. | `$INFINI_RT_ROOT` |
| `INFINI_OPS_SMOKE_BUILD` | Build only the smoke-test operator subset. | `OFF` |
| `INFINI_OPS_OPS` | Comma- or semicolon-separated operator allowlist, or a path to an `ops.json` implementation selection. | empty |
| `INFINI_OPS_TORCH_OPS` | Comma- or semicolon-separated ATen operator allowlist. | empty |

An `ops.json` file selects operators and implementation slots with a top-level
operator mapping:

```json
{
  "add": {
    "implementations": "all"
  },
  "argmax": {
    "implementations": [8]
  },
  "top_k_top_p_sampling_from_logits": {
    "implementations": [16]
  }
}
```

`"all"` keeps every available implementation for the operator. An integer
array keeps exactly those slots. Slots range from 0 through 31. The selection
is a set, not a priority order; the default dispatch selects the smallest
active slot. The selection controls generated wrappers, generated slot-8 ATen
implementations, and linked
provider resolution. Unselected linked providers do not require their external
libraries to be installed.

Pass the file explicitly with
`-DINFINI_OPS_OPS=/path/to/ops.json`. For compatibility,
`${PROJECT_SOURCE_DIR}/ops.json` is read automatically when present. Relative
implementation header paths in legacy configurations are resolved from
`${PROJECT_SOURCE_DIR}`. An explicit inline `INFINI_OPS_OPS` allowlist takes
precedence over an implicit `${PROJECT_SOURCE_DIR}/ops.json`. When
`INFINI_OPS_TORCH_OPS` and an explicit JSON selection are both set, generated
ATen ops use their intersection. The string and string-array values supported
by the current generator remain available for implementation headers that
already exist when CMake configures. Structured descriptors preserve an
explicit backend name, including for implementations outside the standard
backend directory layout. These explicit-header forms remain supported:

```json
{
  "add": "src/native/cpu/ops/add/add.h",
  "gemm": ["src/native/cpu/ops/gemm/gemm.h"],
  "custom_add": [
    {
      "path": "custom/add.h",
      "backend": "custom"
    }
  ]
}
```

Paths to build-generated implementation headers, such as files under
`generated/`, are not supported. Select generated implementations by slot
instead.

Only one GPU backend should be enabled in a build. CPU may be enabled with the
selected accelerator backend.

## Python Wheel Build

Using CPU as the smallest backend:

```bash
python -m pip install .[dev] \
  --config-settings=cmake.define.INFINI_RT_ROOT=/path/to/infini-rt-prefix \
  --config-settings=cmake.define.WITH_CPU=ON
```

Using NVIDIA as an example accelerator backend:

```bash
python -m pip install .[dev] \
  --config-settings=cmake.define.INFINI_RT_ROOT=/path/to/infini-rt-prefix \
  --config-settings=cmake.define.WITH_CPU=ON \
  --config-settings=cmake.define.WITH_NVIDIA=ON
```

Install the Triton JIT runtime dependencies, then enable the implementation:

```bash
python -m pip install torch triton
python -m pip install . \
  --config-settings=cmake.define.INFINI_RT_ROOT=/path/to/infini-rt-prefix \
  --config-settings=cmake.define.WITH_NVIDIA=ON \
  --config-settings=cmake.define.WITH_TRITON=ON
```

The InfiniOps JIT bridge and kernel sources are packaged only with the Python
wheel. It requires CUDA Toolkit 12.0 or newer. Standalone C++ installations do
not provide this runtime.

Compiled kernels are cached in the platform cache directory. Set
`INFINI_OPS_TRITON_CACHE_DIR` to override that location.

Python calls with an explicit Triton config construct an operator for that
call instead of entering the generic operator cache. The compiled kernel and
auto-tuning result are still cached using the complete Triton config
identity.

Full builds with both `WITH_NVIDIA=ON` and `WITH_LINKED=ON` include
`flash_attn_with_kvcache` and require a compatible FlashAttention Python
distribution in the build environment. The distribution must provide a
`flash_attn_2_cuda` shared library that exports the required C++ ABI for the
active PyTorch and CUDA toolchain. Operator-pruned builds require it only when
`flash_attn_with_kvcache` is selected; the default smoke subset does not select
this operator.

The built wheel installs the InfiniOps Python extension and the InfiniRT shared
library needed by the extension.

## Smoke Build

For routine development and pull requests, start with a smoke build:

```bash
python -m pip install .[dev] --no-build-isolation --no-deps \
  --config-settings=cmake.define.INFINI_RT_ROOT=/path/to/infini-rt-prefix \
  --config-settings=cmake.define.WITH_CPU=ON \
  --config-settings=cmake.define.INFINI_OPS_SMOKE_BUILD=ON
```

`INFINI_OPS_SMOKE_BUILD=ON` narrows generated wrappers, bindings, and generated
Torch ops to a representative operator subset. Use full builds for release
preparation, shared build or dispatch changes, wrapper generation changes, and
platform maintainer spot checks.

## Tests

Run the full test suite:

```bash
python -m pytest
```

Run the smoke set:

```bash
python -m pytest tests -m smoke -q
```

Select platforms explicitly:

```bash
python -m pytest tests -m smoke -q --devices cpu nvidia
```

The platform names accepted by the test harness include `nvidia`, `metax`,
`iluvatar`, `hygon`, `moore`, `cambricon`, and `ascend`. The harness maps those
names to the corresponding PyTorch device type when needed.

## Formatting

Run the checks that match the touched files:

```bash
ruff format --check .
ruff check .
```

C++ changes should also pass the repository `clang-format` and `clang-tidy`
expectations described in `CONTRIBUTING.md`.

## Documentation

Enable the Doxygen documentation target with:

```bash
cmake -S . -B build \
  -DINFINI_RT_ROOT=/path/to/infini-rt-prefix \
  -DWITH_CPU=ON \
  -DINFINI_OPS_BUILD_DOCS=ON
cmake --build build --target infiniops_docs
```

The generated HTML is written to `build/docs/reference/html`.

The Documentation Pages workflow uses the same target to validate pull requests
and publish `master` builds through GitHub Pages.

See [API Reference](api/reference.md) for reference scope and preview commands.
