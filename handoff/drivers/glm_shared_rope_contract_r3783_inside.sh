#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export PYTHONPYCACHEPREFIX="/local/rope-prefix-${SLURM_JOB_ID}-${SLURM_STEP_ID}"
COMMON=/refactor/candidates/rebased-5f91ad7-r3668/Common/GLM/InfiniLM
SDK=/refactor/install/hygon-glm-5f91ad7-r3672
ABI=$(python -c 'import torch;print(int(torch.compiled_with_cxx11_abi()))')
TORCH_LIB_DIR=$(python -c 'import pathlib,torch;print(pathlib.Path(torch.__file__).parent/"lib")')
export LD_LIBRARY_PATH="$SDK/lib:$COMMON/python:$COMMON/python/infinilm/_lib:$TORCH_LIB_DIR:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:$LD_LIBRARY_PATH" OMP_NUM_THREADS=1
BASE=/refactor/candidates/glm-shared-rope-component-r3783
mkdir "$BASE"
g++ -std=c++17 -O2 -shared -fPIC -pthread -D_GLIBCXX_USE_CXX11_ABI="$ABI" -I"$COMMON/csrc" -I/refactor/incoming -I"$SDK/include" -I"$COMMON/third_party/json/include" -I"$COMMON/third_party/spdlog/include" /refactor/incoming/glm_shared_rope_contract_r3783.cpp -L"$COMMON/python" -l:_infinilm.cpython-310-x86_64-linux-gnu.so -L"$COMMON/python/infinilm/_lib" -linfinilm_runtime -L"$SDK/lib" -linfiniops -linfinirt -linfiniccl -L/opt/dtk/cuda/cuda/lib64 -lcudart -o "$BASE/probe.so"
export PYTHONPATH="$COMMON/python:/old/offline-deps:${PYTHONPATH:-}"
python - <<'PY'
import torch,infinilm,ctypes
lib=ctypes.CDLL('/refactor/candidates/glm-shared-rope-component-r3783/probe.so');f=lib.glm_shared_rope_contract_r3783;f.argtypes=[ctypes.c_char_p];assert f(b'/refactor/results/glm-shared-rope-component-r3783.json')==0
PY
echo GLM_SHARED_ROPE_R3783_COMPLETE
