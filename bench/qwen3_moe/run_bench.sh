#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 客户端并发吞吐压测（vllm bench serve），对齐 T2-1-2 的服务测法。
# 遍历 concurrency × input-len × output-len 矩阵，每组结果落一个文件。
#
# 前置：serve.sh 已在同机 PORT 上把服务起好。
#
# 用法：
#   TAG=base ./run_bench.sh        # 跑基线版（如 main 的 naive MoE），结果存 bench_results/base
#   TAG=this ./run_bench.sh        # 跑本方案（T2-1-3 grouped_gemm），结果存 bench_results/this
#
# 环境变量：
#   TAG          结果子目录名（base / this），用于 base-vs-this 对比
#   PORT         服务端口（默认 8102）
#   MODEL_NAME   请求里的 model 字段（服务单模型，一般被忽略；默认见下）
#   TOKENIZER    分词器路径，用于准确统计 token 数
#   BATCH_SIZES  concurrency 列表（默认 "1 8 32"）
#   INPUT_LENS   输入长度列表（默认 "32 256 2048"）
#   OUTPUT_LENS  输出长度列表（默认 "256 1024"）
#   REPEAT       每组的请求数 = concurrency × REPEAT（默认 3）
# ---------------------------------------------------------------------------
set -euo pipefail

TAG="${TAG:-this}"
PORT="${PORT:-8102}"
MODEL_NAME="${MODEL_NAME:-Qwen3-30B-A3B-Thinking-2507}"
TOKENIZER="${TOKENIZER:-/data/huggingface_home/Qwen3-30B-A3B-Thinking-2507}"
REPEAT="${REPEAT:-3}"

read -r -a BATCH_SIZES <<< "${BATCH_SIZES:-1 8 32}"
read -r -a INPUT_LENS  <<< "${INPUT_LENS:-32 256 2048}"
read -r -a OUTPUT_LENS <<< "${OUTPUT_LENS:-256 1024}"

OUTDIR="${OUTDIR:-bench_results/${TAG}}"
mkdir -p "${OUTDIR}"
unset http_proxy https_proxy all_proxy ALL_PROXY 2>/dev/null || true

echo ">>> TAG=${TAG} port=${PORT} -> ${OUTDIR}"
echo ">>> concurrency=[${BATCH_SIZES[*]}] input=[${INPUT_LENS[*]}] output=[${OUTPUT_LENS[*]}]"

for bs in "${BATCH_SIZES[@]}"; do
  for il in "${INPUT_LENS[@]}"; do
    for ol in "${OUTPUT_LENS[@]}"; do
      sleep 1
      seed=$(date +%s)
      num_prompts=$(( bs * REPEAT ))
      out="${OUTDIR}/bs${bs}_in${il}_out${ol}.txt"
      echo ">>> bs=${bs} in=${il} out=${ol} num_prompts=${num_prompts} -> ${out}"

      # 用 --extra-body 把 max_tokens/ignore_eos 塞进 chat 请求，保证每请求生成满 output-len，
      # 吞吐才可比（与 T2-1-2 一致）。--request-rate inf = 爆发模式。
      vllm bench serve \
        --backend openai-chat \
        --model "${MODEL_NAME}" \
        --tokenizer "${TOKENIZER}" \
        --endpoint /v1/chat/completions \
        --port "${PORT}" \
        --request-rate inf \
        --seed "${seed}" \
        --num-prompts "${num_prompts}" \
        --max-concurrency "${bs}" \
        --random-input-len "${il}" \
        --extra-body "{\"max_tokens\": ${ol}, \"ignore_eos\": true}" \
        > "${out}" 2>&1 || { echo "!! 该组失败，见 ${out}"; continue; }
    done
  done
done

echo ">>> done. 汇总： python3 $(dirname "$0")/summarize.py ${OUTDIR}"
