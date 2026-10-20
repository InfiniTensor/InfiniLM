#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export PYTHONPYCACHEPREFIX="/local/infini-pycache-${SLURM_JOB_ID}-${SLURM_STEP_ID}-${SLURM_PROCID}"
export ROCBLAS_TENSILE_LIBPATH=/opt/dtk/lib/rocblas/library_gpu5
LM=/refactor/candidates/rebased-5f91ad7-r3668/Common/GLM/InfiniLM
SDK=/refactor/install/hygon-glm-5f91ad7-r3672
export TORCH_LIB_DIR="$(python -c 'import pathlib,torch;print(pathlib.Path(torch.__file__).parent / "lib")')"
export LD_LIBRARY_PATH="$SDK/lib:$LM/python:$LM/python/infinilm/_lib:/refactor/candidates/glm-shared-rope-5f91ad7-r3783/helpers:$TORCH_LIB_DIR:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="$LM/python:/old/offline-deps:${PYTHONPATH:-}"
export LD_PRELOAD=/refactor/candidates/blas-policy-r271/libblas_policy_r271.so
unset RDMAV_DRIVERS NCCL_ALGO NCCL_PROTO NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS
export OMP_NUM_THREADS=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_SOCKET_IFNAME=ib NCCL_IB_DISABLE=0
if [ "$GLM_NETWORK_TRIAL" = shca ]; then
 PLUGIN=/refactor/candidates/public-shca-layers-r2228/layer3-selected/opt
 export LD_LIBRARY_PATH="$PLUGIN/rccl-shca-net/lib:$PLUGIN/ucx-shca/lib:/refactor/candidates/ccl-loader-r1248/rdma:/refactor/build-deps/rdma-host-stack:$LD_LIBRARY_PATH"
 export RDMAV_DRIVERS=/refactor/build-deps/rdma-host-stack/libshca NCCL_NET_PLUGIN=shca NCCL_NET=IBext_v8 UCX_MODULE_DIR="$PLUGIN/ucx-shca/lib/ucx"
 export UCX_NET_DEVICES=shca_0:1,shca_1:1,shca_2:1,shca_3:1 UCX_TLS=rc,sm,self
else
 test "$GLM_NETWORK_TRIAL" = Socket
 export NCCL_NET=Socket
 unset NCCL_NET_PLUGIN UCX_MODULE_DIR UCX_TLS UCX_NET_DEVICES
fi
echo "GLM_NETWORK_R3573 actualrequested=$GLM_NETWORK_TRIAL interface=$NCCL_SOCKET_IFNAME backend=$NCCL_NET"

if [ "$SLURM_PROCID" = 0 ]; then python -u /refactor/incoming/build_glm_latest_cases_r2130.py; fi
timeout --kill-after=20s 28000 python -u /refactor/incoming/modern_glm_shared_rope_r3784.py
