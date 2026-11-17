# Build InfiniLM for MetaX MACA

These instructions build the pinned InfiniRT, InfiniOps, and InfiniCCL stack into one prefix, then build InfiniLM against that prefix. They assume a MACA SDK with `mxcc`, CMake, Ninja, Python 3.10, and a MetaX C550 device.

Set the common environment in the build container:

```bash
export MACA_PATH=/opt/maca
export INFINI_ROOT=/absolute/path/to/coherent-prefix
export INFINILM_CXX11_ABI=1
```

The installed MetaX PyTorch uses C++11 ABI 1. Keep InfiniRT, InfiniOps, InfiniCCL, and InfiniLM on the same ABI.

## InfiniRT

```bash
cmake -S third_party/InfiniCore/submodules/InfiniRT \
  -B build/metax/rt \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$INFINI_ROOT" \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DWITH_CPU=ON \
  -DWITH_METAX=ON \
  -DINFINI_RT_BUILD_TESTING=ON \
  -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=1
cmake --build build/metax/rt -j32
ctest --test-dir build/metax/rt --output-on-failure
cmake --install build/metax/rt
```

## InfiniOps

The profile in `scripts/configs/infiniops_ops_metax_infinilm_native.json` exports every public operator referenced by InfiniLM. Operators that do not yet have an active MetaX implementation are declared `public_api_only`; they are available to compile InfiniLM, but a model that executes one of them reports that no active implementation exists.

```bash
cmake -S third_party/InfiniCore/submodules/InfiniOps \
  -B build/metax/ops \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$INFINI_ROOT" \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DINFINI_RT_ROOT="$INFINI_ROOT" \
  -DWITH_CPU=ON \
  -DWITH_METAX=ON \
  -DWITH_TORCH=OFF \
  -DWITH_LINKED=OFF \
  -DINFINI_OPS_OPS="$PWD/scripts/configs/infiniops_ops_metax_infinilm_native.json" \
  -DCMAKE_C_COMPILER=third_party/InfiniCore/submodules/InfiniOps/scripts/mxcc_wrapper.sh \
  -DCMAKE_CXX_COMPILER=third_party/InfiniCore/submodules/InfiniOps/scripts/mxcc_wrapper.sh \
  -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=1 \
  -DCMAKE_EXE_LINKER_FLAGS=-fopenmp \
  -DCMAKE_SHARED_LINKER_FLAGS=-fopenmp
cmake --build build/metax/ops -j32
cmake --install build/metax/ops
```

## InfiniCCL and InfiniLM

Build InfiniCCL with `WITH_METAX=ON` and `WITH_MCCL=ON`, install it to `$INFINI_ROOT`, then install InfiniLM editable:

```bash
xmake f -y -m release --cxx11-abi=1
python -m pip install -e . --no-build-isolation
```
