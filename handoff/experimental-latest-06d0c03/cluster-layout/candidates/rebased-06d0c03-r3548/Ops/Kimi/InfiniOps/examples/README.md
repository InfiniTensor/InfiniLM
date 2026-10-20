# InfiniOps Examples

This directory contains small examples for development and documentation. They
show how current Python bindings and in-tree C++ development APIs are used.

## Python

Run the Python GEMM example after installing InfiniOps:

```bash
python examples/gemm.py
```

`gemm.py` creates CPU tensors with PyTorch, calls `infini.ops.gemm`, and prints
the result next to `torch.mm`.

## C++

C++ examples are part of the raw CMake build when Python binding generation is
disabled:

```bash
cmake -S . -B build \
  -DINFINI_RT_ROOT=/path/to/infini-rt-prefix \
  -DWITH_CPU=ON \
  -DGENERATE_PYTHON_BINDINGS=OFF
cmake --build build -j
```

The examples are development examples, so they include in-tree headers and link
against the in-tree `infiniops` target. They are not installed downstream
consumer examples.

Build one target explicitly:

```bash
cmake --build build --target tensor
```

## Files

- `tensor.cc`: creates and prints a tensor view.
- `data_type.cc`: prints known data type names and element sizes.
- `gemm.py`: calls the Python `gemm` binding.
- `gemm/`: contains C++ GEMM examples.
- `runtime_api.h`: selects the runtime helper and GEMM implementation headers
  for the enabled backend.
