#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
export PYTHONPYCACHEPREFIX=/local/modal-full-draft-r4487-${SLURM_JOB_ID}-${SLURM_STEP_ID}
export PYTHONPATH=/old/offline-deps:${PYTHONPATH:-} OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -u /refactor/incoming/modal_projection_control_r4487.py
