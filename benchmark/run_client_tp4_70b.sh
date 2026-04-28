#!/usr/bin/env bash
# Variant of run_client.sh tuned for the tp=4 70B server: smaller batch sweep,
# shorter output_len because per-token wall time is ~10x slower than 8B.

set -e

unset http_proxy https_proxy all_proxy ALL_PROXY

OUT_DIR="${OUT_DIR:-$(dirname "$0")/results}"
mkdir -p "${OUT_DIR}"

PORT="${PORT:-8103}"
MODEL_ID="${MODEL_ID:-fm9g_70b_qwen2}"
TOKENIZER="${TOKENIZER:-/data-aisoft/mechdancer/models/FM9G_70B_SFT_MHA_qwen2}"

batch_size_list=(1 4 16)
random_input_len_list=(256)
random_output_len_list=(128)

seed=42
prompts_per_concurrency=10

for batch_size in "${batch_size_list[@]}"; do
    for input_len in "${random_input_len_list[@]}"; do
        for output_len in "${random_output_len_list[@]}"; do
            num_prompts=$(( batch_size * prompts_per_concurrency ))
            if [ "${num_prompts}" -lt 100 ]; then
                num_prompts=100
            fi

            tag="bs=${batch_size}_in=${input_len}_out=${output_len}_n=${num_prompts}_seed=${seed}"
            json_file="${OUT_DIR}/infinilm_A100x4_model=FM9G_70B_qwen2_${tag}.json"

            echo "==========================================="
            echo "tp=4  bs=${batch_size}  in=${input_len}  out=${output_len}  n=${num_prompts}"
            echo "==========================================="

            python "$(dirname "$0")/bench_client.py" \
                --tokenizer "${TOKENIZER}" \
                --model "${MODEL_ID}" \
                --port "${PORT}" \
                --seed "${seed}" \
                --num-prompts "${num_prompts}" \
                --max-concurrency "${batch_size}" \
                --random-input-len "${input_len}" \
                --random-output-len "${output_len}" \
                --ignore-eos \
                > "${json_file}" 2>&1 || true

            tail -n 8 "${json_file}"
            echo
        done
    done
done

echo "All TP=4 70B benchmarks done; JSON results under ${OUT_DIR}"
