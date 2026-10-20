#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export PYTHONPYCACHEPREFIX="/local/glm-cap-${SLURM_JOB_ID}-${SLURM_STEP_ID}"
COMMON=/refactor/candidates/rebased-5f91ad7-r3668/Common/GLM/InfiniLM
SDK=/refactor/install/hygon-glm-5f91ad7-r3672
BASE=/refactor/candidates/glm-capcache-r3806
test -s "$BASE/libglm_capcache_r3806.so"
TORCH_LIB_DIR=$(python -c 'import pathlib,torch;print(pathlib.Path(torch.__file__).parent/"lib")')
export LD_LIBRARY_PATH="$BASE:$SDK/lib:$COMMON/python:$COMMON/python/infinilm/_lib:$TORCH_LIB_DIR:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:$LD_LIBRARY_PATH" OMP_NUM_THREADS=1
export PYTHONPATH="$COMMON/python:/old/offline-deps:${PYTHONPATH:-}"
python -u /refactor/incoming/probe_glm_capcache_r3806.py
