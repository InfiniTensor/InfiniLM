#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export PYTHONPYCACHEPREFIX="/local/kda-round-${SLURM_JOB_ID}-${SLURM_STEP_ID}-${SLURM_PROCID}"
BASE=/refactor/candidates/kda-roundstep-06d-r3655
OPS=/refactor/candidates/rebased-06d0c03-r3548/Ops/Kimi/InfiniOps
test -f "$BASE/libkda_rounding_r3655.so"
export LD_LIBRARY_PATH="/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:$LD_LIBRARY_PATH"
export OMP_NUM_THREADS=1
timeout --kill-after=20s 1200 python -u /refactor/incoming/probe_kda_rounding_r3657.py
