#!/bin/bash
set -e
exec >/home/wuwei/InfiniLM/.tmp-dsv4/inference_server_fused_w13_nondet1285.log 2>&1
echo FUSED_W13_LAUNCH
cd /home/wuwei/InfiniLM
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=/home/wuwei/InfiniLM/python:/home/wuwei/pepe-env-20261016/backup/InfiniCore/python
export CUSTOM_TRANSFORMER_ROOT=/vllm-workspace/vllm-ascend/csrc/build/_CPack_Packages/Linux/External/cann-ops-transformer-custom_linux-aarch64.run/packages/vendors/custom_transformer
export ASCEND_CUSTOM_OPP_PATH="$CUSTOM_TRANSFORMER_ROOT"
export LD_LIBRARY_PATH=/home/wuwei/InfiniLM/.infini-dsv4/lib:"$CUSTOM_TRANSFORMER_ROOT"/op_api/lib:"$LD_LIBRARY_PATH"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export INFINILM_SINGLE_PREFILL_STEP=0
export INFINILM_DYNAMIC_SINGLE_PREFILL_MIN_TOKENS=1024
export INFINILM_DISABLE_MTP=1
export INFINILM_MAX_NUM_BATCHED_TOKENS=4096
# Captured graph inputs are refreshed on every replay.
export INFINILM_DECODE_GRAPH_BATCHES=1,2,4,8,16
export HCCL_DETERMINISTIC=false
export HCCL_OP_EXPANSION_MODE=AIV
exec python python/infinilm/server/inference_server.py --model=/home/wuwei/models/Eco-Tech_DeepSeek-V4-Flash-w8a8-mtp --tp=8 --device=ascend --enable-paged-attn --enable-graph --attn=flash-attn --use-mla --num-draft-tokens=0 --disable-prefix-caching --num-blocks=1285 --block-size=32 --max-batch-size=16 --max-new-tokens=4096 --temperature=0 --top-k=1 --top-p=1 --moe-ep-backend=disabled --tool-call-parser=deepseek_v4 --host=0.0.0.0 --port=18081
