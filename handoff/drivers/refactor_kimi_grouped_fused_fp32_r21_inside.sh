#!/bin/bash
set -euo pipefail
BASE=/refactor/src/InfiniLM
LM=$BASE
SDK=/refactor/install/hygon-runtime-fixes-r1
OPS=/refactor/candidates/runtime-fixes-r1/InfiniOps
ABI=$(python -c 'import torch;print(int(torch.compiled_with_cxx11_abi()))')
export LD_LIBRARY_PATH="$SDK/lib:$LM/python:$LM/python/infinilm/_lib:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:$LD_LIBRARY_PATH"
mkdir -p /refactor/candidates/kimi-grouped-fused-fp32-r21/lib
/opt/dtk/cuda/cuda/bin/nvcc -std=c++17 -O3 -arch=gfx936 -Xcompiler=-fPIC -D_GLIBCXX_USE_CXX11_ABI="$ABI" -c /refactor/incoming/modern_kimi_grouped_fused_fp32_kernels_r21.cu -I"$OPS/src" -o /refactor/candidates/kimi-grouped-fused-fp32-r21/lib/grouped_kernels.o
g++ -std=c++17 -O2 -shared -fPIC -D_GLIBCXX_USE_CXX11_ABI="$ABI" /refactor/incoming/kimi_grouped_fused_fp32_runtime_r21.cpp /refactor/candidates/kimi-grouped-fused-fp32-r21/lib/grouped_kernels.o \
 -I"$BASE/csrc" -I"$SDK/include" -I"$BASE/third_party/spdlog/include" -I/opt/dtk/cuda/cuda/include \
 -L"$BASE/python/infinilm/_lib" -linfinilm_runtime -L"$SDK/lib" -linfiniops -linfinirt -L/opt/dtk/cuda/cuda/lib64 -lcudart \
 -Wl,--no-undefined -o /refactor/candidates/kimi-grouped-fused-fp32-r21/lib/libkimi_grouped_prefill.so
timeout --kill-after=15s 1800 python -u /refactor/incoming/modern_kimi_grouped_fused_fp32_r21.py
