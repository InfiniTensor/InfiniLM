# Build and Test

InfiniRT uses CMake and C++17.

## Common Options

```bash
-DWITH_CPU=ON
-DWITH_NVIDIA=ON
-DWITH_ILUVATAR=ON
-DWITH_METAX=ON
-DWITH_MARS=ON
-DWITH_MOORE=ON
-DWITH_HYGON=ON
-DWITH_CAMBRICON=ON
-DWITH_ASCEND=ON
-DAUTO_DETECT_DEVICES=ON
-DINFINI_RT_BUILD_TESTING=ON
-DINFINI_RT_BUILD_PERFORMANCE_TESTING=ON
-DINFINI_RT_BUILD_DOCS=ON
```

Only one GPU backend can be enabled in a build. CPU can be enabled together
with a GPU backend. If no backend option is enabled, CPU is enabled by default.

## CPU Build

```bash
cmake -S . -B build \
  -DCMAKE_INSTALL_PREFIX=/path/to/infini-rt-prefix \
  -DWITH_CPU=ON \
  -DINFINI_RT_BUILD_TESTING=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
cmake --install build
```

The CPU backend requires OpenMP.

## NVIDIA Build

```bash
cmake -S . -B build \
  -DCMAKE_INSTALL_PREFIX=/path/to/infini-rt-prefix \
  -DWITH_CPU=ON \
  -DWITH_NVIDIA=ON \
  -DINFINI_RT_BUILD_TESTING=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
cmake --install build
```

The NVIDIA backend requires the CUDA toolkit. The installed public headers may
include CUDA headers, so consumers also need access to the CUDA include
directory when compiling against an NVIDIA build.

## Generated Public Headers

During CMake configuration, InfiniRT generates backend-specific public headers
under `generated/include`. The generated headers match the backend options used
for that build and are installed with the hand-written headers.

The recommended consumer include remains:

```cpp
#include <infini/rt.h>
```

## Tests

Enable tests with:

```bash
-DINFINI_RT_BUILD_TESTING=ON
```

Run them with:

```bash
ctest --test-dir build --output-on-failure
```

The `test_install_consumer` test installs InfiniRT to a temporary prefix, uses
`find_package(InfiniRT CONFIG REQUIRED)` from a separate CMake project, links
that project to `InfiniRT::infinirt`, and runs it.

## Performance Tests

Performance tests are built separately from functional tests:

```bash
cmake -S . -B build-perf \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_CPU=ON \
  -DINFINI_RT_BUILD_PERFORMANCE_TESTING=ON
cmake --build build-perf -j
ctest --test-dir build-perf -L performance --output-on-failure
```

Collect JSON results with:

```bash
python3 scripts/run_performance_tests.py \
  --build-dir build-perf \
  --backend cpu \
  --output perf-current.json
```

Compare two local runs with:

```bash
python3 scripts/compare_performance_results.py \
  --baseline perf-baseline.json \
  --candidate perf-current.json
```

Each result reports `unit`, `mean`, and `median` as separate fields. The default
memory sweep runs up to 16 MiB; set `INFINI_RT_PERF_ENABLE_LARGE=1` to include
the 256 MiB case.

## Documentation

Enable the Doxygen documentation target with:

```bash
cmake -S . -B build -DINFINI_RT_BUILD_DOCS=ON
cmake --build build --target infinirt_docs
```

The generated HTML is written to `build/docs/reference/html` under the source
tree.

To preview the generated structure, serve the HTML directory:

```bash
python3 -m http.server 8000 --directory build/docs/reference/html
```

The Documentation Pages workflow uses the same `infinirt_docs` target. Pull
requests build and upload the generated HTML as a Pages artifact for
validation; pushes to `master` deploy it to GitHub Pages.
