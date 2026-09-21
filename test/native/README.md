# Native Regression Tests

These tests exercise ownership, concurrency, configuration parsing, and
paged-attention metadata validation. They do not load model weights or launch
attention kernels.

Run from a Linux development environment with a C++17 compiler, initialized
`third_party` submodules, a matching installed Infini stack, and a freshly built
`infinicore_runtime` target:

```shell
export INFINI_ROOT=/path/to/infini/prefix
bash test/native/run_tests.sh
```

`INFINILM_RUNTIME_DIR` overrides `build/linux/x86_64/release`.
`NATIVE_TEST_BUILD_DIR` overrides `build/native-tests`. The optional argument is
`analyzer`, `runtime`, `paged_attention`, or `config`; the default is `all`.
The analyzer test needs only the compiler and repository headers.

Use the same SDK include paths, C++ ABI, and dynamic-library search paths as the
runtime build. `CXX`, `CXXFLAGS`, and `LDFLAGS` are supported. If the installed
InfiniOps stack links a Python extension provider, standalone executables also
need that provider's Python symbols, for example:

```shell
export LDFLAGS="$(python3-config --embed --ldflags)"
```

Sanitizers apply to the test code and source files compiled by the runner.
Instrument `infinicore_runtime` separately for sanitizer coverage inside the
shared library:

```shell
SANITIZER=thread bash test/native/run_tests.sh analyzer
SANITIZER=address bash test/native/run_tests.sh
```

CUDA initialization can conflict with AddressSanitizer's protected shadow gap
and report `out of memory` even during device enumeration. In that environment,
run address-sanitized tests with `ASAN_OPTIONS=protect_shadow_gap=0`.
