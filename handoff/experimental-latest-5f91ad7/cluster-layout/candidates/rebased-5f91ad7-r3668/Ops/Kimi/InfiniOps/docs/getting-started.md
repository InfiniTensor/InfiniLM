# Getting Started

This guide installs InfiniOps and runs a minimal Python operator call. InfiniOps
depends on an installed InfiniRT prefix.

## Prerequisites

- C++17 compatible compiler
- CMake 3.18 or newer
- Python 3.10 or newer
- A backend SDK for the target device, such as CUDA Toolkit, DTK, MUSA Toolkit,
  Cambricon Neuware, or Ascend CANN
- An installed InfiniRT prefix containing `include/infini/rt.h` and
  `lib/libinfinirt.so`

## Install InfiniRT First

Build and install InfiniRT into a prefix outside the InfiniOps source tree. The
exact backend options should match the device you want InfiniOps to use.

```bash
cmake -S /path/to/InfiniRT -B /tmp/infinirt-build \
  -DCMAKE_INSTALL_PREFIX=/path/to/infini-rt-prefix \
  -DWITH_CPU=ON
cmake --build /tmp/infinirt-build -j
cmake --install /tmp/infinirt-build
```

The prefix should contain:

```text
/path/to/infini-rt-prefix/include/infini/rt.h
/path/to/infini-rt-prefix/lib/libinfinirt.so
```

## Install InfiniOps

For a CPU build:

```bash
python -m pip install . \
  --config-settings=cmake.define.INFINI_RT_ROOT=/path/to/infini-rt-prefix \
  --config-settings=cmake.define.WITH_CPU=ON
```

For development, install the optional dependencies:

```bash
python -m pip install .[dev] \
  --config-settings=cmake.define.INFINI_RT_ROOT=/path/to/infini-rt-prefix \
  --config-settings=cmake.define.WITH_CPU=ON
```

InfiniOps can also auto-detect available backends:

```bash
python -m pip install .[dev] \
  --config-settings=cmake.define.INFINI_RT_ROOT=/path/to/infini-rt-prefix
```

Use explicit backend options when auto-detection is not appropriate for the
target machine.

## Minimal Python Call

```python
import infini.ops
import torch

m, n, k = 2, 3, 4

x = torch.randn(m, k, device="cpu")
y = torch.randn(k, n, device="cpu")
z = torch.empty(m, n, device="cpu")

infini.ops.gemm(x, y, z)

print(z)
```

The repository also contains `examples/gemm.py` for a runnable Python example.
See [Examples](examples.md) for the full example inventory.

## Minimal C++ Include

The formal C++ API entry point is:

```cpp
#include <infini/ops.h>
```

This header exposes the documented core types and generated operator call
surface. Downstream C++ code should use this installed entry point. Repository
examples may use narrower headers or implementation details to exercise specific
components; see [Examples](examples.md) for build commands and backend notes. An
installed-consumer CMake example would add packaging validation, but is not a
prerequisite for treating the documented C++ interface as public API.

## Next Steps

See [Build and Test](build.md) for backend options and smoke test commands, and
[Operators](api/operators.md) for the operator dispatch model.
