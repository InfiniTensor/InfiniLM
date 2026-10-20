#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
NAME=$(printf '%s' "$LATEST_VARIANT_R3508" | tr '[:upper:]' '[:lower:]')
export TORCH_LIB_DIR="$(python -c 'import pathlib,torch;print(pathlib.Path(torch.__file__).parent / "lib")')"
export LD_LIBRARY_PATH="/refactor/install/hygon-${NAME}-c3deba4-r3491/lib:$TORCH_LIB_DIR:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:$LD_LIBRARY_PATH"
export ROCBLAS_TENSILE_LIBPATH=/opt/dtk/lib/rocblas/library_gpu5
unset LD_PRELOAD
export OMP_NUM_THREADS=1
python -u /refactor/incoming/sdk_gpu_load_r3508.py
