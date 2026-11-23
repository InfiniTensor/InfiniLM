#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODEL_PATH=${DSV4_MODEL_PATH:-/home/wuwei/models/Eco-Tech_DeepSeek-V4-Flash-w8a8-mtp}
CUSTOM_TRANSFORMER_ROOT=${INFINI_ASCEND_CUSTOM_OPAPI_ROOT:-/vllm-workspace/vllm-ascend/csrc/build/_CPack_Packages/Linux/External/cann-ops-transformer-custom_linux-aarch64.run/packages/vendors/custom_transformer}
LOG_PATH=${VLLM_LOG_PATH:-$SCRIPT_DIR/vllm_bench_server.log}

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export VLLM_USE_V1=1
export LD_LIBRARY_PATH="$CUSTOM_TRANSFORMER_ROOT/op_api/lib:/usr/local/Ascend/nnal/atb/9.1.0/atb/cxx_abi_0/lib:/usr/local/Ascend/cann-9.1.0/aarch64-linux/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH:-}"
export ASCEND_CUSTOM_OPP_PATH=$CUSTOM_TRANSFORMER_ROOT
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
cd "$SCRIPT_DIR"
exec vllm serve "$MODEL_PATH" --host 0.0.0.0 --port 18082 --trust-remote-code --max-model-len 8448 --tensor-parallel-size 8 --max-num-seqs 16 --gpu-memory-utilization 0.92 >"$LOG_PATH" 2>&1
