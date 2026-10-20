#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
export PYTHONPYCACHEPREFIX="/local/dspark-${SLURM_JOB_ID}-${SLURM_STEP_ID}-${SLURM_PROCID}"
export PYTHONPATH="/old/offline-deps:${PYTHONPATH:-}" OMP_NUM_THREADS=1 MALLOC_ARENA_MAX=8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export LD_PRELOAD=/refactor/candidates/blas-policy-r271/libblas_policy_r271.so
unset SPECFORGE_DRAFT_FLEX_BACKEND NCCL_ALGO NCCL_PROTO NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS
timeout --kill-after=20s 3600 python -u /refactor/incoming/run_dspark_batch_r3640.py
