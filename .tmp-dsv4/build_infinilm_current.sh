#!/usr/bin/env bash
set -euo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export XMAKE_ROOT=y
export PYBIND11_INCLUDE_DIR=/usr/local/python3.12.13/lib/python3.12/site-packages/pybind11/include
export INFINI_ROOT=/home/wuwei/InfiniLM/.infini-dsv4
export INFINI_ASCEND_CUSTOM_OPAPI_ROOT=/vllm-workspace/vllm-ascend/csrc/build/_CPack_Packages/Linux/External/cann-ops-transformer-custom_linux-aarch64.run/packages/vendors/custom_transformer
export INFINILM_CXX11_ABI=0
cd /home/wuwei/InfiniLM
/root/.local/bin/xmake --root -y -F xmake_compat.lua _infinilm
