#!/usr/bin/env bash
# Two-process fused MoE kernel microbench (InfiniLM vendored op vs vLLM wheel).
# Usage:
#   REPO=/path/to/minicpm5-moe-support MODEL=/path/to/checkpoint [TUNED=/path/to/configs] bash run_microbench_fused_moe_two_process.sh
set -euo pipefail
REPO="${REPO:?set REPO to repo root}"
MODEL="${MODEL:?set MODEL to HF checkpoint dir with config.json}"
SWEEP="${SWEEP:-1,8,32,128,512}"
OUT="${OUT:-${REPO}/InfiniLM/examples/bench_artifacts/microbench_fused_moe_kernel.jsonl}"
mkdir -p "$(dirname "$OUT")"

export PYTHONPATH="${REPO}/InfiniLM/python:${REPO}/InfiniCore/python:${PYTHONPATH:-}"
TUNED_ARGS=()
if [[ -n "${TUNED:-}" ]]; then
  TUNED_ARGS=(--tuned-config-dir "$TUNED")
fi

echo "# infinilm side ($(command -v python3))" >&2
python3 "${REPO}/InfiniLM/examples/microbench_fused_moe_kernel.py" --impl infinilm --nvidia \
  --model-path "$MODEL" --num-tokens-sweep "$SWEEP" --seed "${SEED:-0}" \
  --jsonl-out "$OUT" --print-json "${TUNED_ARGS[@]}"

if [[ ! -f "${REPO}/.venv-vllm/bin/python" ]]; then
  echo "skip vllm side: ${REPO}/.venv-vllm/bin/python missing" >&2
  exit 0
fi

echo "# vllm side (.venv-vllm)" >&2
# shellcheck disable=SC1091
source "${REPO}/.venv-vllm/bin/activate"
python "${REPO}/InfiniLM/examples/microbench_fused_moe_kernel.py" --impl vllm --nvidia \
  --model-path "$MODEL" --num-tokens-sweep "$SWEEP" --seed "${SEED:-0}" \
  --jsonl-out "$OUT" --print-json "${TUNED_ARGS[@]}"

echo "Appended rows to $OUT" >&2
