#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export ROCBLAS_TENSILE_LIBPATH=/opt/dtk/lib/rocblas/library_gpu5
LM=/refactor/candidates/rebased-3e0d89fe-r2842/Common/Deep/InfiniLM
SDK=/refactor/install/hygon-deep-cast64-r3315
export LD_PRELOAD=/refactor/candidates/blas-policy-r271/libblas_policy_r271.so
export LD_LIBRARY_PATH="/refactor/candidates/models-3e0d89fe-r2856/Deep:/refactor/candidates/models-3e0d89fe-r2856/Deep/helpers:$SDK/lib:$LM/python:$LM/python/infinilm/_lib:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="$LM/python:/old/offline-deps:${PYTHONPATH:-}"
unset NCCL_ALGO NCCL_PROTO NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS
export OMP_NUM_THREADS=1
python -u /refactor/incoming/build_deep_serving_cases_r2130.py
timeout --kill-after=20s 14400 python -u /refactor/incoming/modern_deep_attention_f64_r3496.py
