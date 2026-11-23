#!/usr/bin/env bash
set -euo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh

CUSTOM_TRANSFORMER_ROOT=/vllm-workspace/vllm-ascend/csrc/build/_CPack_Packages/Linux/External/cann-ops-transformer-custom_linux-aarch64.run/packages/vendors/custom_transformer
export INFINI_ASCEND_CUSTOM_OPAPI_ROOT="${CUSTOM_TRANSFORMER_ROOT}"
export ASCEND_CUSTOM_OPP_PATH="${CUSTOM_TRANSFORMER_ROOT}:${ASCEND_CUSTOM_OPP_PATH:-}"
export LD_LIBRARY_PATH="${CUSTOM_TRANSFORMER_ROOT}/op_api/lib:${LD_LIBRARY_PATH:-}"

cmake -S /home/wuwei/InfiniLM/third_party/InfiniCore/submodules/InfiniOps \
  -B /home/wuwei/InfiniLM/.build-dsv4/InfiniOps \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/home/wuwei/InfiniLM/.infini-dsv4 \
  -DWITH_ASCEND=ON \
  -DGENERATE_PYTHON_BINDINGS=ON \
  -DINFINI_RT_ROOT=/home/wuwei/InfiniLM/.infini-dsv4
cmake --build /home/wuwei/InfiniLM/.build-dsv4/InfiniOps -j16
cmake --install /home/wuwei/InfiniLM/.build-dsv4/InfiniOps
