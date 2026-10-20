#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export PYTHONPYCACHEPREFIX="/local/legacy-pycache-${SLURM_JOB_ID}-${SLURM_STEP_ID}-${SLURM_PROCID}"
export ROCBLAS_TENSILE_LIBPATH=/opt/dtk/lib/rocblas/library_gpu5
LM=/refactor/candidates/kimi-trace-r3/InfiniLM
SDK=/refactor/install/hygon-kimi-static-r330
export LD_LIBRARY_PATH="/refactor/candidates/kimi-grouped-fused-fp32-r21/lib:$SDK/lib:$LM/python:$LM/python/infinilm/_lib:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="$LM/python:/old/offline-deps:${PYTHONPATH:-}"
export LD_PRELOAD=/refactor/candidates/blas-policy-r271/libblas_policy_r271.so
unset NCCL_ALGO NCCL_PROTO NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS
export OMP_NUM_THREADS=1 NCCL_DEBUG=WARN NCCL_IB_DISABLE=0
if [ "$SLURM_PROCID" = 0 ]; then python -u /refactor/incoming/build_kimi_serving_cases_r1101.py; fi
python -c 'import os,pathlib,time; p=pathlib.Path("/refactor/results/kimi-serving-cases-r1101-"+os.environ["SLURM_JOB_ID"]+"."+os.environ["SLURM_STEP_ID"]+".json"); start=time.monotonic(); exec("while not p.exists():\n assert time.monotonic()-start<1200; time.sleep(.1)")'
timeout --kill-after=30s 18000 python -u /refactor/incoming/legacy_kimi_eos_r3472.py
