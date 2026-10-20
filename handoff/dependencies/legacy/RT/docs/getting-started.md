# Getting Started

This guide installs InfiniRT and builds a small application against the
installed headers and library.

## Build and Install

Choose an install prefix outside the source tree:

```bash
cmake -S . -B build \
  -DCMAKE_INSTALL_PREFIX=/path/to/infini-rt-prefix \
  -DWITH_CPU=ON
cmake --build build -j
cmake --install build
```

The install prefix contains:

```text
/path/to/infini-rt-prefix/include
/path/to/infini-rt-prefix/lib
```

On platforms that install libraries to `lib64`, use that directory in place of
`lib`.

## Minimal Consumer

Use the single public entry header:

```cpp
#include <cstddef>

#include <infini/rt.h>

int main() {
  namespace rt = infini::rt::runtime;

  rt::SetDevice(0);

  void* ptr = nullptr;
  constexpr std::size_t size = 1024;

  if (rt::Malloc(&ptr, size) != rt::kSuccess) {
    return 1;
  }
  if (rt::Memset(ptr, 0, size) != rt::kSuccess) {
    rt::Free(ptr);
    return 1;
  }
  if (rt::Free(ptr) != rt::kSuccess) {
    return 1;
  }

  return 0;
}
```

Create a `CMakeLists.txt` that imports the installed package:

```cmake
cmake_minimum_required(VERSION 3.18)
project(infinirt_example LANGUAGES CXX)

find_package(InfiniRT 0.1 CONFIG REQUIRED)

add_executable(infinirt_example main.cc)
target_link_libraries(infinirt_example PRIVATE InfiniRT::infinirt)
```

Configure and build it against the installed prefix:

```bash
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH=/path/to/infini-rt-prefix
cmake --build build -j
```

The `InfiniRT::infinirt` target supplies the required include path, library,
C++ language level, and backend dependencies. For an accelerator build, the
corresponding SDK must also be available when configuring the consumer.

## Next Steps

See [Runtime API](api/runtime.md) for backend dispatch and runtime functions,
and [Build and Test](build.md) for backend-specific build options.
