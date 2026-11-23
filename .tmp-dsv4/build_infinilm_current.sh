#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
INSTALL_ROOT=${INFINI_ROOT:-$REPO_ROOT/.infini-dsv4}
CUSTOM_TRANSFORMER_ROOT=${INFINI_ASCEND_CUSTOM_OPAPI_ROOT:-/vllm-workspace/vllm-ascend/csrc/build/_CPack_Packages/Linux/External/cann-ops-transformer-custom_linux-aarch64.run/packages/vendors/custom_transformer}
XMAKE_BIN=${XMAKE_BIN:-/root/.local/bin/xmake}

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export XMAKE_ROOT=y
export PYBIND11_INCLUDE_DIR=${PYBIND11_INCLUDE_DIR:-/usr/local/python3.12.13/lib/python3.12/site-packages/pybind11/include}
export INFINI_ROOT=$INSTALL_ROOT
export INFINI_ASCEND_CUSTOM_OPAPI_ROOT=$CUSTOM_TRANSFORMER_ROOT
export INFINILM_CXX11_ABI=0
cd "$REPO_ROOT"
"$XMAKE_BIN" --root -y -F xmake_compat.lua _infinilm
