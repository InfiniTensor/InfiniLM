#!/usr/bin/env bash
# InfiniLM inference server, adapted from run_infinilm_server_benchmark.sh
# for the post-refactor (CMake) tree.
#
# Original differences vs this version:
#   - cwd was /home/wangpengcheng/workspace_test/InfiniLM (xmake-driven)
#   - now /home/zhangyue/restruct/InfiniLM (CMake-driven)
#   - underscored CLI args replaced with dashed equivalents per base_config.py
#   - --nvidia replaced with --device nvidia

set -e

GPU="${GPU:-0}"
PORT="${PORT:-8102}"
ROOT="${ROOT:-/home/zhangyue/restruct/InfiniLM}"
MODEL="${MODEL:-/data-aisoft/mechdancer/models/9g_8b_thinking_llama/}"
TP="${TP:-1}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-64}"
NUM_BLOCKS="${NUM_BLOCKS:-1024}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"

export CUDA_VISIBLE_DEVICES="${GPU}"

# Incremental build with flash-attn enabled (the original benchmark uses --attn flash-attn)
FA_DIR="${FA_DIR:-${ROOT}/third_party/flash-attention}"
FA_ARCHS="${FA_ARCHS:-80}"
if [ ! -f "${FA_DIR}/csrc/flash_attn/flash_api.cpp" ]; then
    git clone --depth 1 https://github.com/vllm-project/flash-attention.git "${FA_DIR}"
fi
if [ ! -f "${FA_DIR}/csrc/cutlass/include/cutlass/cutlass.h" ]; then
    git -C "${FA_DIR}" submodule update --init --recursive
fi

cmake -S "${ROOT}" -B "${ROOT}/build" -DCMAKE_BUILD_TYPE=Release \
    -DINFINIOPS_FLASH_ATTN_DIR="${FA_DIR}" \
    -DINFINIOPS_FLASH_ATTN_ARCHS="${FA_ARCHS}"
cmake --build "${ROOT}/build" -j"$(nproc)"

cd "${ROOT}"

exec python python/infinilm/server/inference_server.py \
    --device nvidia \
    --model "${MODEL}" \
    --temperature 1.0 \
    --top-p 0.8 \
    --top-k 1 \
    --port "${PORT}" \
    --tp "${TP}" \
    --block-size 256 \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --num-blocks "${NUM_BLOCKS}" \
    --max-batch-size "${MAX_BATCH_SIZE}" \
    --enable-graph \
    --cache-type paged \
    --attn flash-attn
