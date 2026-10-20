#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export PYTHONPYCACHEPREFIX="/local/infini-pycache-${SLURM_JOB_ID}-${SLURM_STEP_ID}-${SLURM_PROCID}"
export ROCBLAS_TENSILE_LIBPATH=/opt/dtk/lib/rocblas/library_gpu5
LM=/refactor/candidates/rebased-3e0d89fe-r2842/Common/GLM/InfiniLM
SDK=/refactor/install/hygon-glm-3e0d89fe-r2844
export TORCH_LIB_DIR="$(python -c 'import pathlib,torch;print(pathlib.Path(torch.__file__).parent / "lib")')"
export LD_LIBRARY_PATH="$SDK/lib:$LM/python:$LM/python/infinilm/_lib:/refactor/candidates/glm-expert-order-r3348/helpers:$TORCH_LIB_DIR:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="$LM/python:/old/offline-deps:${PYTHONPATH:-}"
export LD_PRELOAD=/refactor/candidates/blas-policy-r271/libblas_policy_r271.so
unset RDMAV_DRIVERS NCCL_ALGO NCCL_PROTO NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS
export OMP_NUM_THREADS=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
if [ "$SLURM_PROCID" = 0 ]; then python -u /refactor/incoming/build_glm_latest_cases_r2130.py; fi
timeout --kill-after=20s 21000 python -u /refactor/incoming/modern_glm_expert_order_r3353.py
