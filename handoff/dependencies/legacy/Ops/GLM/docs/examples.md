# Examples

InfiniOps keeps small runnable examples under `examples/`. They are intended
for development and smoke validation of common API surfaces, not as installed
downstream consumer projects.

## Python Example

Run the Python GEMM example after installing the InfiniOps wheel:

```bash
python examples/gemm.py
```

The example creates CPU tensors with PyTorch, calls `infini.ops.gemm`, and
prints the InfiniOps result next to `torch.mm`.

## C++ Examples

C++ examples are built by the raw CMake project when Python binding generation
is disabled:

```bash
cmake -S . -B build \
  -DINFINI_RT_ROOT=/path/to/infini-rt-prefix \
  -DWITH_CPU=ON \
  -DGENERATE_PYTHON_BINDINGS=OFF
cmake --build build -j
```

The generated executables are written under the CMake build tree. Exact paths
depend on the generator, but common targets include:

- `tensor`
- `data_type`
- `gemm`
- `gemm_dispatch`

Build one example target explicitly:

```bash
cmake --build build --target gemm
```

## Example Inventory

| Example | Language | Purpose | Backend notes |
| --- | --- | --- | --- |
| `examples/gemm.py` | Python | Calls `infini.ops.gemm` on PyTorch tensors. | CPU by default. |
| `examples/tensor.cc` | C++ | Creates and prints an `infini::ops::Tensor` view. | CPU host memory only. |
| `examples/data_type.cc` | C++ | Prints data type names and element sizes. | Backend independent. |
| `examples/gemm/gemm.cc` | C++ | Allocates backend memory and calls `Gemm::Call`. | Uses the enabled backend selected by `examples/runtime_api.h`. |
| `examples/gemm/gemm_dispatch.cc` | C++ | Compares two NVIDIA GEMM implementation indexes. | Requires `WITH_NVIDIA=ON`; exits early on other builds. |

## Scope

The C++ examples are repository-local validation programs, not the definition of
the public API boundary. They may include narrower headers such as `tensor.h`,
`runtime_api.h`, or backend implementation headers to exercise specific
components. Downstream C++ code should start with `<infini/ops.h>` and use the
documented API surface. A separate installed-consumer CMake example would be
useful for validating packaging and header discovery.
