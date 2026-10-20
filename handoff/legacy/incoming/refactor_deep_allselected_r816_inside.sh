#!/bin/bash
set -euo pipefail
LM=/refactor/src/InfiniLM
SDK=/refactor/install/hygon-v4-indexer-ops-r1
export LD_PRELOAD=/refactor/candidates/blas-policy-r271/libblas_policy_r271.so
export LD_LIBRARY_PATH="/refactor/candidates/native-engine-allselected-r815:/refactor/build:$SDK/lib:$LM/python:$LM/python/infinilm/_lib:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="$LM/python:/old/offline-deps:${PYTHONPATH:-}"
unset NCCL_ALGO NCCL_PROTO NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS
export OMP_NUM_THREADS=1
timeout --kill-after=20s 14400 python -u /refactor/incoming/modern_deep_allselected_r816.py
