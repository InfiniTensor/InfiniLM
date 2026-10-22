#!/bin/bash
set -euo pipefail
BASE=/refactor/candidates/deep-collapse-5f91ad7-r4367
COMMON=/refactor/candidates/rebased-5f91ad7-r3668/Common/Deep/InfiniLM
SDK=/refactor/install/hygon-deep-5f91ad7-r3672
export LD_LIBRARY_PATH="$COMMON/python:$COMMON/python/infinilm/_lib:$SDK/lib:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
ABI=$(python -c 'import torch;print(int(torch.compiled_with_cxx11_abi()))')
test ! -e "$BASE"
mkdir "$BASE"
cp /refactor/incoming/deep_collapse_r4367.cu /refactor/incoming/deep_collapse_probe_r4367.cpp /refactor/incoming/modern_deep_collapse_r4367.py /refactor/incoming/deep_collapse_r4367_inside.sh "$BASE/"
/opt/dtk/cuda/cuda/bin/nvcc -std=c++17 -O3 --fmad=false -arch=gfx936 -Xcompiler=-fPIC -D_GLIBCXX_USE_CXX11_ABI="$ABI" -c "$BASE/deep_collapse_r4367.cu" -o "$BASE/collapse.o"
/opt/dtk/cuda/cuda/bin/nvcc -std=c++17 -O3 --fmad=false -arch=gfx936 -Xcompiler=-fPIC -D_GLIBCXX_USE_CXX11_ABI="$ABI" -c /refactor/incoming/deepseek_mhc_post_fused_r135.cu -o "$BASE/compare.o"
g++ -std=c++17 -O2 -shared -fPIC -pthread -D_GLIBCXX_USE_CXX11_ABI="$ABI" "$BASE/deep_collapse_probe_r4367.cpp" "$BASE/collapse.o" "$BASE/compare.o" -I"$COMMON/csrc" -I"$SDK/include" -I"$SDK/include/infiniccl" -I"$COMMON/third_party/json/include" -I"$COMMON/third_party/spdlog/include" -I/opt/dtk/cuda/cuda/include -L"$COMMON/python" -l:_infinilm.cpython-310-x86_64-linux-gnu.so -L"$COMMON/python/infinilm/_lib" -linfinilm_runtime -L"$SDK/lib" -linfiniops -linfinirt -L/opt/dtk/cuda/cuda/lib64 -lcudart -Wl,-Bsymbolic -Wl,--no-undefined -o "$BASE/libmodern_deep_collapse_r4367.so"
python -u "$BASE/modern_deep_collapse_r4367.py"
