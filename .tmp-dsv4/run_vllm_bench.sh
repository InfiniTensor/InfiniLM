#!/usr/bin/env bash
set -u

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODEL_NAME=${VLLM_MODEL_NAME:-Eco-Tech_DeepSeek-V4-Flash-w8a8-mtp}
echo "VLLM_DRIVER_START $(date -Is)"
for spec in '4 1024 1024' '4 4096 4096' '16 128 128' '16 1024 1024' '16 4096 4096'; do
  set -- $spec
  python3 "$SCRIPT_DIR/bench_latency.py" --url http://127.0.0.1:18082 --engine vllm --model "$MODEL_NAME" --conc "$1" --input "$2" --output "$3" --timeout 21600
done
echo "VLLM_DRIVER_DONE $(date -Is)"
