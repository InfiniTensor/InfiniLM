# Backends

InfiniOps can be built for CPU and one accelerator backend at a time. The
Python and operator APIs remain common, while device SDKs, compiler flags, and
available implementations differ by backend.

## Backend Options

| Backend | CMake option | Notes |
| --- | --- | --- |
| CPU | `WITH_CPU` | Used as the smallest build and can be enabled with one accelerator backend. |
| NVIDIA | `WITH_NVIDIA` | Requires CUDA Toolkit. |
| Iluvatar | `WITH_ILUVATAR` | CUDA-compatible backend using the CoreX toolchain. |
| Hygon | `WITH_HYGON` | Requires DTK. `DTK_ROOT` defaults to `/opt/dtk` when unset. |
| T-Head | `WITH_THEAD` | Uses the T-Head CUDA-compatible SDK from `CUDA_SDK`, defaulting to `/usr/local/PPU_SDK/CUDA_SDK`. |
| MetaX | `WITH_METAX` | Requires the MetaX runtime and SDK paths. |
| Cambricon | `WITH_CAMBRICON` | Requires Cambricon Neuware. |
| Moore | `WITH_MOORE` | Requires MUSA Toolkit through `MUSA_ROOT`, `MUSA_HOME`, `MUSA_PATH`, or `/usr/local/musa`. |
| Ascend | `WITH_ASCEND` | Requires Ascend CANN and, by default, the custom AscendC kernel toolchain. |
| PyTorch C++ | `WITH_TORCH` | Adds ATen-backed implementations when PyTorch C++ headers and libraries are available. |

## Device Auto-Detection

`AUTO_DETECT_DEVICES=ON` probes device files such as `/dev/nvidia*` and turns on
matching backend options. This is useful on configured developer machines but
can be too implicit for reproducible CI or release builds.

Prefer explicit backend options in scripts, CI, and release instructions.

## Backend Selection in Tests

The Python test harness accepts platform names through `--devices`, for example:

```bash
python -m pytest tests -m smoke -q --devices cpu nvidia
```

Supported selector names include:

- `nvidia`
- `metax`
- `iluvatar`
- `hygon`
- `thead`
- `moore`
- `cambricon`
- `ascend`

The harness maps those platform names to the PyTorch device type used by the
installed backend, such as `cuda`, `musa`, `mlu`, or `npu`.

## Implementation Layout

Backend implementations live under:

```text
src/native/<category>/<platform>/ops/<op>/
```

Examples include:

- `src/native/cpu/ops/gemm/`
- `src/native/cuda/nvidia/ops/gemm/`
- `src/native/ascend/ops/matmul/`
- `src/native/cambricon/ops/rms_norm/`

The PyTorch C++ backend uses:

```text
src/torch/ops/<op>/
generated/torch/<op>/
```

Generated files are build artifacts and should not be edited by hand.
