#!/bin/bash
set -euo pipefail
LM=/refactor/candidates/kimi-trace-r3/InfiniLM
SDK=/refactor/install/hygon-kimi-static-r330
export LD_LIBRARY_PATH="/refactor/candidates/kimi-grouped-fused-fp32-r21/lib:$SDK/lib:$LM/python:$LM/python/infinilm/_lib:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="$LM/python:/old/offline-deps:${PYTHONPATH:-}"
export LD_PRELOAD=/refactor/candidates/blas-policy-r271/libblas_policy_r271.so
unset NCCL_ALGO NCCL_PROTO NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS
export OMP_NUM_THREADS=1 NCCL_DEBUG=WARN NCCL_IB_DISABLE=0
timeout --kill-after=30s 13800 python -u /refactor/incoming/modern_kimi_token_only_r964.py
