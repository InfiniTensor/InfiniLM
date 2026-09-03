#!/bin/bash
# InfiniLM Route B (native GGUF quantization) development environment.
# Source this file after setting CUDA_HOME and optional CUTLASS_ROOT/CUDNN_ROOT.
ROUTEB_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROUTEB_INFINILM_DIR="$(cd -- "${ROUTEB_SCRIPT_DIR}/.." && pwd)"
: "${INFINICORE_DIR:=$(cd -- "${ROUTEB_INFINILM_DIR}/../InfiniCore" && pwd)}"
: "${INFINI_ROOT:=${HOME}/.infini}"

if [[ -n "${CUDA_HOME:-}" ]]; then
    export CUDACXX="${CUDACXX:-${CUDA_HOME}/bin/nvcc}"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    ROUTEB_CUDA_LIB="${CUDA_HOME}/lib64:"
else
    ROUTEB_CUDA_LIB=""
fi

export INFINICORE_DIR INFINI_ROOT
export PYTHONPATH="${INFINICORE_DIR}/python:${ROUTEB_INFINILM_DIR}/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${INFINICORE_DIR}/python/infinicore/lib:${ROUTEB_INFINILM_DIR}/python/infinilm/lib:${INFINI_ROOT}/lib:${ROUTEB_CUDA_LIB}${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
