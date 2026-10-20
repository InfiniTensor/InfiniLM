#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
export PYTHONPYCACHEPREFIX="/local/infini-pycache-${SLURM_JOB_ID}-${SLURM_STEP_ID}-${SLURM_PROCID}"
export ROCBLAS_TENSILE_LIBPATH=/opt/dtk/lib/rocblas/library_gpu5
test -f "$ROCBLAS_TENSILE_LIBPATH/TensileLibrary_lazy_gfx936.dat"
export PATH=/root/.local/bin:/opt/dtk/llvm/bin:/opt/dtk/bin:${PATH}
SOURCE=/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM
SDK=/refactor/install/hygon-kimi-5f91ad7-r3672
export TORCH_LIB_DIR="$(python -c 'import pathlib,torch;print(pathlib.Path(torch.__file__).parent / "lib")')"
export PYTHONPATH="$SOURCE/python:/old/offline-deps:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/refactor/candidates/models-5f91ad7-r3675/Kimi/helpers:$SDK/lib:$SOURCE/python:$SOURCE/python/infinilm/_lib:$TORCH_LIB_DIR:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:$LD_LIBRARY_PATH"
export LD_PRELOAD=/refactor/candidates/blas-policy-r271/libblas_policy_r271.so
unset RDMAV_DRIVERS NCCL_ALGO NCCL_PROTO NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS INFINILM_KIMI_TRACE_DIR INFINILM_KIMI_KDA_TRACE_DIR INFINILM_KIMI_ROUTE_TAG_R585 INFINILM_KIMI_ROUTE_DIR_R585 INFINILM_KIMI_DIAG_DEVICE_SYNC INFINILM_KIMI_DIAG_STREAM_SYNC
export MALLOC_ARENA_MAX=8
export OMP_NUM_THREADS=1 NCCL_DEBUG=INFO NCCL_IB_DISABLE=0 NCCL_SOCKET_IFNAME=ib PYTHONFAULTHANDLER=1
PLUGIN=/refactor/candidates/public-shca-layers-r2228/layer3-selected/opt
export NCCL_DEBUG_SUBSYS=INIT,NET
if [ "$GLOBAL_MODEL_MODE_R2354" = shca ]; then
 export LD_LIBRARY_PATH="$PLUGIN/rccl-shca-net/lib:$PLUGIN/ucx-shca/lib:/refactor/candidates/ccl-loader-r1248/rdma:/refactor/build-deps/rdma-host-stack:$LD_LIBRARY_PATH"
 export RDMAV_DRIVERS=/refactor/build-deps/rdma-host-stack/libshca NCCL_NET_PLUGIN=shca NCCL_NET=IBext_v8 UCX_MODULE_DIR="$PLUGIN/ucx-shca/lib/ucx"
 export UCX_NET_DEVICES=shca_0:1,shca_1:1,shca_2:1,shca_3:1 UCX_TLS=rc,sm,self
else
 test "$GLOBAL_MODEL_MODE_R2354" = Socket
 export NCCL_NET=Socket
 unset NCCL_NET_PLUGIN UCX_MODULE_DIR UCX_TLS UCX_NET_DEVICES
fi
export INFINILM_KIMI_SPEC_R3632=1
timeout --kill-after=10s 180 python -u /refactor/incoming/postcancel_health_r1997.py
if [ "${KIMI_USE_RAMCACHE_R3679:-0}" = 1 ] && [ "$SLURM_PROCID" = 0 ]; then
 CACHE_BASE="/dev/shm/iloader-r3604-$(id -u)"
 export KIMI_MODEL_VIEW_R3679="$CACHE_BASE/overlay-kimi-pp4-s0"
 test -d "$KIMI_MODEL_VIEW_R3679"
 export PYTHONPATH="/refactor/candidates/InfiniLoader-88d6bac-r3599/src:$PYTHONPATH"
 timeout --kill-after=30s 28000 python -m infiniloader run --base-dir "$CACHE_BASE" kimi-pp4-s0 -- python -u /refactor/incoming/modern_kimi_root_r3837.py
else
 timeout --kill-after=30s 28000 python -u /refactor/incoming/modern_kimi_root_r3837.py
fi
