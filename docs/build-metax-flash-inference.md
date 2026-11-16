# Build InfiniLM flash inference for MetaX MACA

Build a separate InfiniOps prefix with the linked-provider profile, `WITH_TORCH=ON`, and `WITH_LINKED=ON`. The profile selects the existing MetaX flash-attn linked providers for prefill and decode; all other operator selections remain the same as the native profile.

```bash
cmake -S third_party/InfiniCore/submodules/InfiniOps \
  -B build/metax/ops-flash \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$INFINI_ROOT" \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DINFINI_RT_ROOT="$INFINI_ROOT" \
  -DWITH_CPU=ON \
  -DWITH_METAX=ON \
  -DWITH_TORCH=ON \
  -DWITH_LINKED=ON \
  -DINFINI_OPS_OPS="$PWD/scripts/configs/infiniops_ops_metax_infinilm_flash.json" \
  -DCMAKE_C_COMPILER=third_party/InfiniCore/submodules/InfiniOps/scripts/mxcc_wrapper.sh \
  -DCMAKE_CXX_COMPILER=third_party/InfiniCore/submodules/InfiniOps/scripts/mxcc_wrapper.sh \
  -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=1 \
  -DCMAKE_EXE_LINKER_FLAGS=-fopenmp \
  -DCMAKE_SHARED_LINKER_FLAGS=-fopenmp
cmake --build build/metax/ops-flash --target infiniops -j32
cmake --install build/metax/ops-flash
```

Install InfiniCCL into the same prefix, then build InfiniLM with the same ABI and `INFINI_ROOT`.

Some MetaX `flash_attn_2_cuda` builds report an allocator corruption error during Python teardown even though generation completed. Keep inference servers process-resident until the vendor extension fixes its exit path.
