#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export PYTHONPYCACHEPREFIX="/local/glm-cap-${SLURM_JOB_ID}-${SLURM_STEP_ID}"
COMMON=/refactor/candidates/rebased-5f91ad7-r3668/Common/GLM/InfiniLM
SDK=/refactor/install/hygon-glm-5f91ad7-r3672
OPS=/refactor/candidates/rebased-5f91ad7-r3668/Ops/GLM/InfiniOps
BASE=/refactor/candidates/glm-capcache-r3806
mkdir "$BASE"
ABI=$(python -c 'import torch;print(int(torch.compiled_with_cxx11_abi()))')
TORCH_LIB_DIR=$(python -c 'import pathlib,torch;print(pathlib.Path(torch.__file__).parent/"lib")')
export LD_LIBRARY_PATH="$BASE:$SDK/lib:$COMMON/python:$COMMON/python/infinilm/_lib:$TORCH_LIB_DIR:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:$LD_LIBRARY_PATH" OMP_NUM_THREADS=1
export PYTHONPATH="$COMMON/python:/old/offline-deps:${PYTHONPATH:-}"
/opt/dtk/cuda/cuda/bin/nvcc -std=c++17 -O3 -arch=gfx936 -Xcompiler=-fPIC -D_GLIBCXX_USE_CXX11_ABI="$ABI" -I"$OPS/src" -I"$SDK/include" -I/refactor/incoming -shared /refactor/incoming/glm_fp8_capplan_r3806.cu /refactor/incoming/glm_fp8_capcache_r3806.cpp -lcublas -o "$BASE/libglm_capcache_r3806.so"
python -u /refactor/incoming/probe_glm_capcache_r3806.py
