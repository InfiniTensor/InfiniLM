#!/usr/bin/env bash
# InfiniLM MiniCPM5 MoE correctness: canonical bench_balanced + MoE backend A/B.
# Run on host or inside minicpm5-moe. Uses prefix LD_PRELOAD (never export LD_PRELOAD).
#
# Pick GPU **inside** the container shell (host-prefixed `CUDA_VISIBLE_DEVICES=1 docker exec …`
# is not reliable for remapping the GPU this process sees):
#
#   docker exec minicpm5-moe bash -lc 'export CUDA_VISIBLE_DEVICES=1; bash /home/zenghua/workspace/minicpm5-moe-support/InfiniLM/examples/run_correctness_bench_smoke.sh'
#
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
REPO="${REPO:-/home/zenghua/workspace/minicpm5-moe-support}"
MODEL="${MODEL:-/data-aisoft/zenghua/models/minicpm5.16a3.v0314}"
export PYTHONPATH="$REPO/InfiniLM/python:$REPO/InfiniCore/python:${PYTHONPATH:-}"
TORCH_LIB="$(python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), \"lib\"))")"
FA="/usr/local/lib/python3.12/dist-packages/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so"
unset LD_LIBRARY_PATH
unset LD_PRELOAD
export LD_LIBRARY_PATH="/root/.infini/lib:$TORCH_LIB:/usr/local/lib/python3.12/dist-packages:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu"
ART="$REPO/InfiniLM/examples/bench_artifacts"
export ART
mkdir -p "$ART"
cd "$REPO/InfiniLM/examples"

run_one() {
  local label="$1"
  local json_name="$2"
  local moe_env="${3:-}"
  echo "=== ${label} ===" >&2
  if [[ -n "$moe_env" ]]; then
    env INFINILM_FORCE_MOE_BACKEND="$moe_env" LD_PRELOAD="$FA" python3 bench_balanced.py --nvidia \
      --model-path "$MODEL" \
      --prompt "Hi" \
      --prompt-tokens 7 \
      --max-new-tokens 4 \
      --top-k 1 \
      --attn flash-attn \
      --enable-paged-attn \
      --paged-kv-block-size 256 \
      --warmup-steps 3 \
      --runs 1 \
      --seed 0 \
      --json-out "$ART/$json_name" \
      --print-json
  else
    LD_PRELOAD="$FA" python3 bench_balanced.py --nvidia \
      --model-path "$MODEL" \
      --prompt "Hi" \
      --prompt-tokens 7 \
      --max-new-tokens 4 \
      --top-k 1 \
      --attn flash-attn \
      --enable-paged-attn \
      --paged-kv-block-size 256 \
      --warmup-steps 3 \
      --runs 1 \
      --seed 0 \
      --json-out "$ART/$json_name" \
      --print-json
  fi
}

run_one "1) Canonical smoke (auto / fused-capable MoE)" "single_prompt_infini_smoke.json" ""
run_one "2) MoE forced vllm_fused" "single_prompt_infini_smoke_vllm_fused.json" "vllm_fused"
run_one "3) MoE forced baseline" "single_prompt_infini_smoke_baseline.json" "baseline"

echo "=== Summary (generated excerpt + TTFT) ===" >&2
python3 -c "
import json, os
art = os.environ['ART']
for name in (
    'single_prompt_infini_smoke.json',
    'single_prompt_infini_smoke_vllm_fused.json',
    'single_prompt_infini_smoke_baseline.json',
):
    p = os.path.join(art, name)
    d = json.load(open(p, encoding='utf-8'))
    gen = d.get('generated_text', '')
    excerpt = (gen[:120] + '…') if len(gen) > 120 else gen
    print(name, 'ttft_ms=', round(d['ttft_ms'], 2), 'gen=', repr(excerpt))
"

echo "DONE" >&2
