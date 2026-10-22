#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
BASE=/refactor/candidates/kimi-expert-batch-5f91ad7-r4391
COMMON=/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM
SDK=/refactor/install/hygon-kimi-5f91ad7-r3672
OPS=/refactor/candidates/rebased-5f91ad7-r3668/Ops/Kimi/InfiniOps
export TORCH_LIB_DIR="$(python -c 'import pathlib,torch;print(pathlib.Path(torch.__file__).parent / "lib")')"
export LD_LIBRARY_PATH="$SDK/lib:$COMMON/python:$COMMON/python/infinilm/_lib:$TORCH_LIB_DIR:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$COMMON/python:/old/offline-deps:${PYTHONPATH:-}"
export PYTHONPYCACHEPREFIX="/local/kimi-expert-batch-${SLURM_JOB_ID}-${SLURM_STEP_ID}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
unset LD_PRELOAD
ABI=$(python -c 'import torch;print(int(torch.compiled_with_cxx11_abi()))')
test ! -e "$BASE"
mkdir "$BASE"
cp /refactor/incoming/kimi_tiled_batch_r4391.cu /refactor/incoming/kimi_expert_native_r4391.cpp /refactor/incoming/modern_kimi_expert_batch_r4391.py /refactor/incoming/kimi_expert_batch_r4391_inside.sh "$BASE/"
cp /refactor/incoming/kimi_mxfp4_static_r326.cuh /refactor/incoming/kimi_tiled_w13_r1858.cuh /refactor/incoming/kimi_reduce_wave_r1044.cuh "$BASE/"
cp /refactor/candidates/kimi-dense0-5f91ad7-r4346/tile.o "$BASE/current-model-tile.o"
/opt/dtk/cuda/cuda/bin/nvcc -std=c++17 -O3 -arch=gfx936 -Xcompiler=-fPIC -D_GLIBCXX_USE_CXX11_ABI="$ABI" -I"$BASE" -I"$OPS/src" -I"$OPS/generated" -I"$SDK/include" -c "$BASE/kimi_tiled_batch_r4391.cu" -o "$BASE/batch.o"
g++ -std=c++17 -O2 -shared -fPIC -pthread -D_GLIBCXX_USE_CXX11_ABI="$ABI" "$BASE/kimi_expert_native_r4391.cpp" "$BASE/batch.o" "$BASE/current-model-tile.o" -I"$COMMON/csrc" -I"$SDK/include" -I"$SDK/include/infiniccl" -I/opt/dtk/cuda/cuda/include -L"$COMMON/python" -l:_infinilm.cpython-310-x86_64-linux-gnu.so -L"$COMMON/python/infinilm/_lib" -linfinilm_runtime -L"$SDK/lib" -linfiniops -linfinirt -L/opt/dtk/cuda/cuda/lib64 -lcudart -lcuda -Wl,-Bsymbolic -Wl,--no-undefined -o "$BASE/libmodern_kimi_expert_batch_r4391.so"
python -u "$BASE/modern_kimi_expert_batch_r4391.py"
