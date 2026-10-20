#!/bin/bash
set -euo pipefail
LM=/refactor/src/InfiniLM
SDK=/refactor/install/hygon-fp8-expert-r19
export LD_LIBRARY_PATH="/refactor/build:$SDK/lib:$LM/python:$LM/python/infinilm/_lib:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="$LM/python:/old/offline-deps:${PYTHONPATH:-}"
export LD_PRELOAD=/refactor/candidates/blas-policy-r271/libblas_policy_r271.so
unset NCCL_ALGO NCCL_PROTO NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS
export OMP_NUM_THREADS=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
timeout --kill-after=20s 18000 python -u /refactor/incoming/modern_glm_cache_view_r928.py
