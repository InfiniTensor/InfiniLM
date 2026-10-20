# InfiniOps

InfiniOps is a high-performance, cross-platform operator library supporting multiple backends: CPU, Nvidia, MetaX, Iluvatar, Hygon, Moore, Cambricon, and more.

## Prerequisites

- C++17 compatible compiler
- CMake 3.18+
- Python 3.10+
- Hardware-specific SDKs (e.g. CUDA Toolkit, MUSA Toolkit)
- Installed InfiniRT headers and library

## Installation

Install InfiniRT first, then build InfiniOps with the InfiniRT install prefix:

```bash
pip install . -C cmake.define.INFINI_RT_ROOT=/path/to/infini-rt-prefix
```

`/path/to/infini-rt-prefix` is the directory passed to InfiniRT as
`CMAKE_INSTALL_PREFIX`; it should contain `include/infini/rt.h` and
`lib/libinfinirt.so`.

InfiniOps auto-detects available platforms on supported backends. To specify
platforms explicitly:

```bash
pip install . \
  -C cmake.define.INFINI_RT_ROOT=/path/to/infini-rt-prefix \
  -C cmake.define.WITH_CPU=ON \
  -C cmake.define.WITH_NVIDIA=ON
```

The Python wheel installs the required InfiniRT shared library next to the
InfiniOps extension so `import infini.ops` can load its runtime dependency.

### CMake Options

| Option | Description | Default |
|---|---|:-:|
| `-DWITH_CPU=[ON\|OFF]` | Compile the CPU implementation | OFF |
| `-DWITH_NVIDIA=[ON\|OFF]` | Compile the Nvidia implementation | OFF |
| `-DWITH_METAX=[ON\|OFF]` | Compile the MetaX implementation | OFF |
| `-DWITH_ILUVATAR=[ON\|OFF]` | Compile the Iluvatar implementation | OFF |
| `-DWITH_HYGON=[ON\|OFF]` | Compile the Hygon implementation | OFF |
| `-DWITH_THEAD=[ON\|OFF]` | Compile the T-Head PPU implementation | OFF |
| `-DWITH_MOORE=[ON\|OFF]` | Compile the Moore implementation | OFF |
| `-DWITH_CAMBRICON=[ON\|OFF]` | Compile the Cambricon implementation | OFF |
| `-DWITH_ASCEND=[ON\|OFF]` | Compile the Ascend implementation | OFF |
| `-DWITH_TORCH=[ON\|OFF]` | Compile generated PyTorch ATen-backed operators | OFF |
| `-DAUTO_DETECT_DEVICES=[ON\|OFF]` | Auto-detect available platforms | ON |
| `-DINFINI_RT_ROOT=<path>` | InfiniRT install prefix containing `include/` and `lib/` | `$INFINI_RT_ROOT` |

If no accelerator options are provided and auto-detection finds nothing, `WITH_CPU` is enabled by default.

For Hygon builds, set `DTK_ROOT` to the DTK installation root if it is not installed at `/opt/dtk`. You can override the default DCU arch with `-DHYGON_ARCH=<arch>` when configuring CMake.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for code style, commit conventions, PR workflow, development guide, and troubleshooting.

## Documentation

Full user and contributor documentation starts at [docs/README.md](docs/README.md).

Useful entry points:

- [Getting Started](docs/getting-started.md)
- [Build and Test](docs/build.md)
- [Backends](docs/backends.md)
- [Examples](docs/examples.md)
- [Operators](docs/api/operators.md)
- [Operator API Alignment](docs/operator-api-alignment.md)
- [API Reference](docs/api/reference.md)
- [Adding ATen-backed operators](docs/aten-operators.md)
- [Compatibility](docs/compatibility.md)

## License

See [LICENSE](LICENSE).
