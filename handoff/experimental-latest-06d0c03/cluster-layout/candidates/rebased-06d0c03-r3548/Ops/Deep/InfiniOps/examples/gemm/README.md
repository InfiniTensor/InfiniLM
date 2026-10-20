# GEMM Examples

This directory contains C++ GEMM examples for the in-tree InfiniOps build.

## `gemm.cc`

`gemm.cc` demonstrates the basic C++ development flow:

1. Create host tensors.
2. Allocate backend memory through the selected runtime helper.
3. Copy inputs to the backend.
4. Create backend tensor views.
5. Call `Gemm::Call`.
6. Copy the output back to host memory.

Build it with the raw CMake project:

```bash
cmake -S . -B build \
  -DINFINI_RT_ROOT=/path/to/infini-rt-prefix \
  -DWITH_CPU=ON \
  -DGENERATE_PYTHON_BINDINGS=OFF
cmake --build build --target gemm
```

Enable an accelerator backend, such as `WITH_NVIDIA=ON`, to run the same example
against that backend.

## `gemm_dispatch.cc`

`gemm_dispatch.cc` demonstrates explicit implementation selection with
`Config::set_implementation_index`.

The example compares NVIDIA implementation index `0` and implementation index
`1`, so it requires:

```bash
-DWITH_NVIDIA=ON
```

On non-NVIDIA builds, the executable prints a message and exits successfully
without running the comparison.
