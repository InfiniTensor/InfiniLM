#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export PYTHONPYCACHEPREFIX="/local/legacy-pycache-${SLURM_JOB_ID}-${SLURM_STEP_ID}-${SLURM_PROCID}"
export ROCBLAS_TENSILE_LIBPATH=/opt/dtk/lib/rocblas/library_gpu5
LM=/refactor/src/InfiniLM
SDK=/refactor/install/hygon-fp8-expert-r19
export LD_LIBRARY_PATH="/refactor/build:$SDK/lib:$LM/python:$LM/python/infinilm/_lib:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="$LM/python:/old/offline-deps:${PYTHONPATH:-}"
export LD_PRELOAD=/refactor/candidates/blas-policy-r271/libblas_policy_r271.so
unset NCCL_ALGO NCCL_PROTO NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS
export OMP_NUM_THREADS=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
if [ "$SLURM_PROCID" = 0 ]; then python -u /refactor/incoming/build_glm_serving_cases_r1186.py; fi
timeout --kill-after=20s 21000 python -u /refactor/incoming/legacy_glm_eos_r3429.py
