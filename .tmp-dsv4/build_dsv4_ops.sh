#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
BUILD_ROOT=${DSV4_BUILD_ROOT:-$REPO_ROOT/.build-dsv4/InfiniOps}
INSTALL_ROOT=${INFINI_ROOT:-$REPO_ROOT/.infini-dsv4}
CUSTOM_TRANSFORMER_ROOT=${INFINI_ASCEND_CUSTOM_OPAPI_ROOT:-/vllm-workspace/vllm-ascend/csrc/build/_CPack_Packages/Linux/External/cann-ops-transformer-custom_linux-aarch64.run/packages/vendors/custom_transformer}

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export INFINI_ASCEND_CUSTOM_OPAPI_ROOT=$CUSTOM_TRANSFORMER_ROOT
export ASCEND_CUSTOM_OPP_PATH="$CUSTOM_TRANSFORMER_ROOT:${ASCEND_CUSTOM_OPP_PATH:-}"
export LD_LIBRARY_PATH="$CUSTOM_TRANSFORMER_ROOT/op_api/lib:${LD_LIBRARY_PATH:-}"

cmake -S "$REPO_ROOT/third_party/InfiniCore/submodules/InfiniOps" -B "$BUILD_ROOT" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTALL_ROOT" -DWITH_ASCEND=ON -DGENERATE_PYTHON_BINDINGS=ON -DINFINI_RT_ROOT="$INSTALL_ROOT"
cmake --build "$BUILD_ROOT" --parallel
cmake --install "$BUILD_ROOT"
