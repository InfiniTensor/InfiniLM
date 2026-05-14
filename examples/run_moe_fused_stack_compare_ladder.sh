#!/usr/bin/env bash
# Ordered **vendor vs upstream** ladder: microbench gap → token sweep → concurrency sweep → optional smoke → optional e2e.
# Intended inside `minicpm5-moe` with GPU visible; see `minicpm5_moe_inference_profiling.md` § Concurrency (c>1).
#
#   docker exec minicpm5-moe bash -lc 'export CUDA_VISIBLE_DEVICES=1; bash /home/zenghua/workspace/minicpm5-moe-support/InfiniLM/examples/run_moe_fused_stack_compare_ladder.sh'
#
# Env:
#   SKIP_MICROBENCH_GAP=1
#   SKIP_MICROBENCH_TOKEN_SWEEP=1  — skip `run_moe_fused_stack_microbench_token_sweep.sh` (4× gap runs).
#   SKIP_CONCURRENCY_SWEEP=1       — skip `run_moe_fused_stack_concurrency_sweep.sh` (two server boots × |CONCURRENCIES|).
#   SKIP_SMOKE=1          — do not run `run_correctness_bench_smoke.sh` (longer).
#   SKIP_E2E_COMPARE=1    — do not run `run_e2e_moe_fused_stack_compare.sh` (two server boots).
#   Any env forwarded to nested scripts (e.g. REPO, MODEL, VPY, PORT_VENDOR, CONCURRENCIES, CUDA_VISIBLE_DEVICES).
#
set -euo pipefail

REPO="${REPO:-/home/zenghua/workspace/minicpm5-moe-support}"
export REPO

echo "=== [ladder] MoE fused stack compare: microbench → token sweep → concurrency sweep → smoke → e2e ===" >&2

if [[ "${SKIP_MICROBENCH_GAP:-0}" != "1" ]]; then
  bash "$REPO/InfiniLM/examples/run_moe_fused_stack_microbench_gap.sh"
else
  echo "[ladder] SKIP_MICROBENCH_GAP=1" >&2
fi

if [[ "${SKIP_MICROBENCH_TOKEN_SWEEP:-0}" != "1" ]]; then
  bash "$REPO/InfiniLM/examples/run_moe_fused_stack_microbench_token_sweep.sh"
else
  echo "[ladder] SKIP_MICROBENCH_TOKEN_SWEEP=1" >&2
fi

if [[ "${SKIP_CONCURRENCY_SWEEP:-0}" != "1" ]]; then
  bash "$REPO/InfiniLM/examples/run_moe_fused_stack_concurrency_sweep.sh"
else
  echo "[ladder] SKIP_CONCURRENCY_SWEEP=1" >&2
fi

if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
  bash "$REPO/InfiniLM/examples/run_correctness_bench_smoke.sh"
else
  echo "[ladder] SKIP_SMOKE=1" >&2
fi

if [[ "${SKIP_E2E_COMPARE:-0}" != "1" ]]; then
  bash "$REPO/InfiniLM/examples/run_e2e_moe_fused_stack_compare.sh"
else
  echo "[ladder] SKIP_E2E_COMPARE=1" >&2
fi

echo "[ladder] DONE" >&2
