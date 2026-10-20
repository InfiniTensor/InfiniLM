#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export PYTHONPYCACHEPREFIX="/local/kimi-smoke-${SLURM_JOB_ID}-${SLURM_STEP_ID}"
BASE=/refactor/candidates/kimi-sdk-smoke-5f91ad7-r3714
COMMON=/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM
SDK=/refactor/install/hygon-kimi-5f91ad7-r3672
mkdir "$BASE"
ABI=$(python -c 'import torch;print(int(torch.compiled_with_cxx11_abi()))')
export TORCH_LIB_DIR="$(python -c 'import pathlib,torch;print(pathlib.Path(torch.__file__).parent/"lib")')"
export LD_LIBRARY_PATH="$SDK/lib:$COMMON/python:$COMMON/python/infinilm/_lib:$TORCH_LIB_DIR:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="$COMMON/python:/old/offline-deps:${PYTHONPATH:-}" OMP_NUM_THREADS=1 INFINILM_OWNER_CPU_WIDTH_R2392=4
g++ -std=c++17 -O2 -shared -fPIC -pthread -D_GLIBCXX_USE_CXX11_ABI="$ABI" -I"$COMMON/csrc" -I"$SDK/include" -I"$COMMON/third_party/json/include" -I"$COMMON/third_party/spdlog/include" -I/opt/dtk/cuda/cuda/include /refactor/incoming/kimi_sdk_smoke_r3714.cpp -L"$COMMON/python" -l:_infinilm.cpython-310-x86_64-linux-gnu.so -L"$COMMON/python/infinilm/_lib" -linfinilm_runtime -L"$SDK/lib" -linfiniops -linfinirt -L/opt/dtk/cuda/cuda/lib64 -lcudart -Wl,--no-undefined -o "$BASE/libkimi_sdk_smoke_r3714.so"
python -u /refactor/incoming/kimi_sdk_smoke_r3714.py
