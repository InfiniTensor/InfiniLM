#!/bin/bash
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
MODEL_PATH=${DSV4_MODEL_PATH:-/home/wuwei/models/Eco-Tech_DeepSeek-V4-Flash-w8a8-mtp}
INSTALL_ROOT=${INFINI_ROOT:-$REPO_ROOT/.infini-dsv4}
CUSTOM_TRANSFORMER_ROOT=${INFINI_ASCEND_CUSTOM_OPAPI_ROOT:-/vllm-workspace/vllm-ascend/csrc/build/_CPack_Packages/Linux/External/cann-ops-transformer-custom_linux-aarch64.run/packages/vendors/custom_transformer}
LOG_PATH=${INFINILM_LOG_PATH:-$SCRIPT_DIR/inference_server_w8a8_dsv4.log}

exec >"$LOG_PATH" 2>&1
echo W8A8_DSV4_LAUNCH
cd "$REPO_ROOT"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH="$REPO_ROOT/python:/home/wuwei/pepe-env-20261016/backup/InfiniCore/python${PYTHONPATH:+:$PYTHONPATH}"
export CUSTOM_TRANSFORMER_ROOT
export ASCEND_CUSTOM_OPP_PATH="$CUSTOM_TRANSFORMER_ROOT"
export LD_LIBRARY_PATH="$INSTALL_ROOT/lib:$CUSTOM_TRANSFORMER_ROOT/op_api/lib:${LD_LIBRARY_PATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export INFINILM_SINGLE_PREFILL_STEP=0
export INFINILM_DYNAMIC_SINGLE_PREFILL_MIN_TOKENS=1024
export INFINILM_DISABLE_MTP=1
export INFINILM_MAX_NUM_BATCHED_TOKENS=4096
export INFINILM_DECODE_GRAPH_BATCHES=1,2,4,8,16
export HCCL_DETERMINISTIC=false
export HCCL_OP_EXPANSION_MODE=AIV
exec python python/infinilm/server/inference_server.py --model="$MODEL_PATH" --tp=8 --device=ascend --enable-paged-attn --enable-graph --attn=flash-attn --use-mla --num-draft-tokens=0 --disable-prefix-caching --num-blocks=1285 --block-size=32 --max-batch-size=16 --max-new-tokens=4096 --temperature=0 --top-k=1 --top-p=1 --moe-ep-backend=disabled --tool-call-parser=deepseek_v4 --host=0.0.0.0 --port=18081
