#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export PYTHONPYCACHEPREFIX="/local/deep-token-${SLURM_JOB_ID}-${SLURM_STEP_ID}-${SLURM_PROCID}"
export ROCBLAS_TENSILE_LIBPATH=/opt/dtk/lib/rocblas/library_gpu5
LM=/refactor/candidates/rebased-5f91ad7-r3668/Common/Deep/InfiniLM
SDK=/refactor/install/hygon-deep-5f91ad7-r3672
export LD_PRELOAD=/refactor/candidates/blas-policy-r271/libblas_policy_r271.so
export LD_LIBRARY_PATH="/refactor/candidates/models-5f91ad7-r3675/Deep:/refactor/candidates/models-5f91ad7-r3675/Deep/helpers:$SDK/lib:$LM/python:$LM/python/infinilm/_lib:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="$LM/python:/old/offline-deps:${PYTHONPATH:-}"
unset NCCL_ALGO NCCL_PROTO NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS
export OMP_NUM_THREADS=1
python -u /refactor/incoming/build_deep_serving_cases_r2130.py
if [ "${DEEP_USE_RAMCACHE_R3616:-0}" = 1 ]; then
 CACHE_BASE="/dev/shm/iloader-r3607-$(id -u)"
 export DEEP_MODEL_VIEW_R3616="$CACHE_BASE/overlay-deep-pp1-s0"
 test -d "$DEEP_MODEL_VIEW_R3616"
 export PYTHONPATH="/refactor/candidates/InfiniLoader-88d6bac-r3599/src:$PYTHONPATH"
 timeout --kill-after=20s 14400 python -m infiniloader run --base-dir "$CACHE_BASE" deep-pp1-s0 -- python -u /refactor/incoming/modern_deep_attention_precision_r3798.py
else
 timeout --kill-after=20s 14400 python -u /refactor/incoming/modern_deep_attention_precision_r3798.py
fi
