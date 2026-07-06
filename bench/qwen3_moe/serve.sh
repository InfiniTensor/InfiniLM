#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 启动 InfiniLM OpenAI 兼容服务，用于 Qwen3-30B-A3B-Thinking-2507 的并发吞吐基准。
# 目标平台：沐曦 MetaX C500，TP=2（两张 64GB 卡）。
#
# 用法：
#   MODEL=/data/huggingface_home/Qwen3-30B-A3B-Thinking-2507 ./serve.sh
#   MAX_CON=32 NUM_BLOCKS=1024 ./serve.sh      # 覆盖默认值
#   GRAPH=1 ./serve.sh                          # 试开 graph（MoE 含 host dispatch，可能不兼容）
#
# 环境变量（均可覆盖）：
#   MODEL       模型路径
#   PORT        服务端口（默认 8102，与客户端脚本一致）
#   TP          张量并行度（默认 2）
#   MAX_CON     --max-batch-size，必须 >= 你要压测的最大 concurrency（默认 32）
#   NUM_BLOCKS  KV cache 分块数（默认 1024；OOM 就调小，不够就调大，见 README）
#   MAX_NEW     单请求最大生成 token（默认 4096，需 >= 压测的最大 output-len）
#   MAX_CACHE   --max-cache-len（默认 4096，需 >= 最大 input+output）
#   ATTN        注意力后端（默认 paged-attn；MetaX 未编 flash-attn，不要用 flash-attn）
#   PAGED       1=开 paged KV cache（默认，并发批处理需要）；0=退回 static cache
# ---------------------------------------------------------------------------
set -euo pipefail

MODEL="${MODEL:-/data/huggingface_home/Qwen3-30B-A3B-Thinking-2507}"
DEVICE="${DEVICE:-metax}"
PORT="${PORT:-8102}"
TP="${TP:-2}"
MAX_CON="${MAX_CON:-32}"
NUM_BLOCKS="${NUM_BLOCKS:-1024}"
MAX_NEW="${MAX_NEW:-4096}"
MAX_CACHE="${MAX_CACHE:-4096}"
ATTN="${ATTN:-paged-attn}"

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${REPO}/python:${PYTHONPATH:-}"

OPT_FLAGS=()
[[ "${GRAPH:-0}" == "1" ]] && OPT_FLAGS+=(--enable-graph)
[[ "${PAGED:-1}" == "1" ]] && OPT_FLAGS+=(--enable-paged-attn --num-blocks "${NUM_BLOCKS}")

echo ">>> serving ${MODEL}"
echo ">>> device=${DEVICE} tp=${TP} port=${PORT} max_batch=${MAX_CON} attn=${ATTN} paged=${PAGED:-1} num_blocks=${NUM_BLOCKS} graph=${GRAPH:-0}"

exec python3 "${REPO}/python/infinilm/server/inference_server.py" \
    --model "${MODEL}" \
    --device "${DEVICE}" --tp "${TP}" \
    --port "${PORT}" \
    --max-batch-size "${MAX_CON}" \
    --max-new-tokens "${MAX_NEW}" \
    --max-cache-len "${MAX_CACHE}" \
    --temperature 1.0 --top-p 0.8 --top-k 1 \
    --attn "${ATTN}" \
    --ignore-eos \
    "${OPT_FLAGS[@]}"
