#!/usr/bin/env bash
# Run each longtext/decode metric case in a **separate** Python process to release CUDA
# memory between runs (reduces OOM when sweeping 16k/32k/64k × HF + InfiniLM).
#
# Usage (inside minicpm-sala, after picking an idle GPU):
#   export CUDA_VISIBLE_DEVICES=2
#   export NVML_GPU_INDEX=2
#   export REPO=/home/zenghua/workspace/minicpm-sala-support
#   export PYTHONPATH=$REPO/InfiniLM/examples:$REPO/InfiniCore/python:$REPO/InfiniLM/python
#   export LD_LIBRARY_PATH=/root/.infini/lib:${LD_LIBRARY_PATH:-}
#   export METRICS_DATE=2026-03-23
#   cd $REPO/InfiniLM/examples && ./run_longtext_metrics_cases.sh
#
# Optional:
#   METRICS_TARGETS=16384,32768   METRICS_DECODE_STEPS=32  ./run_longtext_metrics_cases.sh
#   SLEEP_BETWEEN_SEC=3   # extra pause between subprocesses

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

REPO="${REPO:-/home/zenghua/workspace/minicpm-sala-support}"
export PYTHONPATH="${SCRIPT_DIR}:${REPO}/InfiniCore/python:${REPO}/InfiniLM/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/root/.infini/lib:${LD_LIBRARY_PATH:-}"

: "${CUDA_VISIBLE_DEVICES:=0}"
: "${NVML_GPU_INDEX:=${CUDA_VISIBLE_DEVICES}}"
: "${METRICS_DATE:=2026-03-23}"
: "${METRICS_DECODE_STEPS:=32}"
: "${METRICS_TARGETS:=16384,32768,65536}"
: "${SLEEP_BETWEEN_SEC:=2}"

OUT_JSONL="${OUT_JSONL:-${SCRIPT_DIR}/profiling_runs/longtext_decode_rows.jsonl}"
mkdir -p "$(dirname "$OUT_JSONL")"
rm -f "$OUT_JSONL"
echo "[run_longtext_metrics] jsonl -> $OUT_JSONL  GPU smi index=$NVML_GPU_INDEX"

IFS=',' read -r -a TARGETS <<< "$METRICS_TARGETS"

run_one() {
  local c="$1"
  echo "[run_longtext_metrics] case=$c"
  python3 collect_metrics_longtext_decode.py --case "$c" --append-jsonl "$OUT_JSONL" || true
  sleep "${SLEEP_BETWEEN_SEC}"
}

for t in "${TARGETS[@]}"; do
  run_one "hf:${t}"
done
for t in "${TARGETS[@]}"; do
  run_one "infinilm_rec:${t}:1"
done
for t in "${TARGETS[@]}"; do
  run_one "infinilm_rec:${t}:${METRICS_DECODE_STEPS}"
done

echo "[run_longtext_metrics] merged table:"
python3 collect_metrics_longtext_decode.py --from-jsonl "$OUT_JSONL"
