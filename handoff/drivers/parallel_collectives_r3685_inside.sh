#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export PYTHONPYCACHEPREFIX="/local/parallel-${SLURM_JOB_ID}-${SLURM_STEP_ID}"
BASE=/refactor/candidates/parallel-collectives-5f91ad7-r3685
LM=/refactor/candidates/rebased-5f91ad7-r3668/Common/GLM/InfiniLM
SDK=/refactor/install/hygon-glm-5f91ad7-r3672
export TORCH_LIB_DIR="$(python -c 'import pathlib,torch;print(pathlib.Path(torch.__file__).parent / "lib")')"
export LD_LIBRARY_PATH="$SDK/lib:$LM/python:$LM/python/infinilm/_lib:$TORCH_LIB_DIR:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="$LM/python:/old/offline-deps:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET NCCL_SOCKET_IFNAME=ib NCCL_IB_DISABLE=0
unset LD_PRELOAD NCCL_ALGO NCCL_PROTO NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS
PLUGIN=/refactor/candidates/public-shca-layers-r2228/layer3-selected/opt
export LD_LIBRARY_PATH="$PLUGIN/rccl-shca-net/lib:$PLUGIN/ucx-shca/lib:/refactor/candidates/ccl-loader-r1248/rdma:/refactor/build-deps/rdma-host-stack:$LD_LIBRARY_PATH"
export RDMAV_DRIVERS=/refactor/build-deps/rdma-host-stack/libshca NCCL_NET_PLUGIN=shca NCCL_NET=IBext_v8 UCX_MODULE_DIR="$PLUGIN/ucx-shca/lib/ucx"
export UCX_NET_DEVICES=shca_0:1,shca_1:1,shca_2:1,shca_3:1 UCX_TLS=rc,sm,self
if [ "$SLURM_PROCID" = 0 ]; then
 mkdir "$BASE"
 ABI=$(python -c 'import torch;print(int(torch.compiled_with_cxx11_abi()))')
 g++ -std=c++17 -O2 -shared -fPIC -pthread -D_GLIBCXX_USE_CXX11_ABI="$ABI" -I"$LM/csrc" -I"$SDK/include" -I"$SDK/include/infiniccl" -I"$LM/third_party/json/include" -I"$LM/third_party/spdlog/include" -I/opt/dtk/cuda/cuda/include /refactor/incoming/parallel_collectives_r3612.cpp -L"$LM/python" -l:_infinilm.cpython-310-x86_64-linux-gnu.so -L"$LM/python/infinilm/_lib" -linfinilm_runtime -L"$SDK/lib" -linfiniops -linfiniccl -linfinirt -L/opt/dtk/cuda/cuda/lib64 -lcudart -Wl,--no-undefined -o "$BASE/libparallel_r3612.so"
 touch "$BASE/build.ready"
fi
for unused in $(seq 1 120); do
 if [ -e "$BASE/build.ready" ]; then break; fi
 sleep 1
done
test -e "$BASE/build.ready"
timeout --kill-after=20s 1500 python -u /refactor/incoming/probe_parallel_collectives_r3685.py
