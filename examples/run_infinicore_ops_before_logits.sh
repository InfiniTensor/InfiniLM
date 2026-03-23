#!/usr/bin/env bash
# InfiniCore CUDA operator smoke tests for MiniCPM-SALA-related ops.
# Run inside minicpm-sala docker before minicpm_sala_logits_sanity.py.
set -euo pipefail

REPO="${REPO:-/home/zenghua/workspace/minicpm-sala-support}"
export PYTHONPATH="$REPO/InfiniCore/test/infinicore:$REPO/InfiniCore/python:$REPO/InfiniLM/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/root/.infini/lib:${LD_LIBRARY_PATH:-}"

OPS_DIR="$REPO/InfiniCore/test/infinicore/ops"
cd "$OPS_DIR"

echo "[run_infinicore_ops] REPO=$REPO"
echo "[run_infinicore_ops] test_infllmv2_attention.py --nvidia"
python3 test_infllmv2_attention.py --nvidia
echo "[run_infinicore_ops] test_simple_gla_prefill.py --nvidia"
python3 test_simple_gla_prefill.py --nvidia
echo "[run_infinicore_ops] OK"
