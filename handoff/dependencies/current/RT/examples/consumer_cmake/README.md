# InfiniRT CMake Consumer Example

This example builds a small executable against an installed InfiniRT prefix.

Install InfiniRT first:

```bash
cmake -S ../.. -B ../../build \
  -DCMAKE_INSTALL_PREFIX=/path/to/infini-rt-prefix \
  -DWITH_CPU=ON
cmake --build ../../build -j
cmake --install ../../build
```

Then build this consumer:

```bash
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH=/path/to/infini-rt-prefix
cmake --build build -j
./build/infinirt_consumer
```

