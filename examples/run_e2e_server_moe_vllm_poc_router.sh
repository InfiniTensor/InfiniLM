#!/usr/bin/env bash
# E2E OpenAI server: vllm_fused + INFINILM_MOE_ROUTER_ENGINE=vllm_poc (imports vLLM grouped_topk).
# Uses $REPO/.venv-vllm/bin/python because system python3 in minicpm5-moe has no vllm.
# Attention is --attn default (venv typically lacks flash_attn). KV is --cache_type static
# (venv transformers + paged hit startup error: invalid static kv cache config type).
#
#   docker exec minicpm5-moe bash -lc 'export CUDA_VISIBLE_DEVICES=1; bash /home/zenghua/workspace/minicpm5-moe-support/InfiniLM/examples/run_e2e_server_moe_vllm_poc_router.sh'
#
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
REPO="${REPO:-/home/zenghua/workspace/minicpm5-moe-support}"
MODEL="${MODEL:-/data-aisoft/zenghua/models/minicpm5.16a3.v0314}"
PORT="${PORT:-8017}"
MODEL_ID="$(basename "$MODEL")"
VPY="${VPY:-$REPO/.venv-vllm/bin/python}"

export PYTHONPATH="$REPO/InfiniLM/python:$REPO/InfiniCore/python:${PYTHONPATH:-}"
export INFINILM_FORCE_MOE_BACKEND=vllm_fused
export INFINILM_MOE_ROUTER_ENGINE=vllm_poc
unset INFINILM_MOE_ROUTER

TORCH_LIB="$("$VPY" -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), \"lib\"))")"
unset LD_LIBRARY_PATH
unset LD_PRELOAD
export LD_LIBRARY_PATH="/root/.infini/lib:$TORCH_LIB:/usr/local/lib/python3.12/dist-packages:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu"

ART="$REPO/InfiniLM/examples/bench_artifacts"
mkdir -p "$ART"
PIDFILE="/tmp/infini_moe_e2e_vllm_poc_${PORT}.pid"
LOGFILE="/tmp/infini_moe_e2e_vllm_poc_${PORT}.log"

"$VPY" -m pip install -q "openai>=1.0" 2>/dev/null || true

cleanup() {
  if [[ -f "$PIDFILE" ]]; then
    kill "$(cat "$PIDFILE")" 2>/dev/null || true
    rm -f "$PIDFILE"
  fi
}
trap cleanup EXIT

free_port() {
  python3 - <<PY || true
import os, socket
port = int("$PORT")
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.bind(("127.0.0.1", port))
    s.close()
except OSError:
    os.system(f"fuser -k {port}/tcp 2>/dev/null || true")
PY
  sleep 2
}

cleanup
free_port
sleep 1

echo "[e2e-vllm-poc] starting server port=$PORT VPY=$VPY INFINILM_MOE_ROUTER_ENGINE=vllm_poc attn=default" >&2
cd "$REPO/InfiniLM/python"
# static KV: venv ships transformers>=5 with vLLM; paged cache hit "invalid static kv cache config type" on startup.
"$VPY" -m infinilm.server.inference_server --nvidia \
  --model_path "$MODEL" \
  --dtype bfloat16 \
  --attn default \
  --cache_type static \
  --num_blocks 256 \
  --block_size 256 \
  --max_batch_size 8 \
  --max_tokens 256 \
  --port "$PORT" \
  --host 127.0.0.1 >"$LOGFILE" 2>&1 &
echo $! >"$PIDFILE"

for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "[e2e-vllm-poc] server ready" >&2
    break
  fi
  if ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "[e2e-vllm-poc] server died during startup" >&2
    tail -120 "$LOGFILE" >&2 || true
    exit 1
  fi
  sleep 2
done

run_perf() {
  local out_json="$1"
  shift
  echo "[e2e-vllm-poc] test_perf -> $out_json $*" >&2
  "$VPY" "$REPO/InfiniLM/scripts/test_perf.py" "$@" --json-out "$out_json"
  "$VPY" -c "
import json, os, sys
p = sys.argv[1]
d = json.load(open(p, encoding='utf-8'))
d['infinilm_moe_backend'] = os.environ.get('INFINILM_FORCE_MOE_BACKEND', '')
d['infinilm_moe_router'] = os.environ.get('INFINILM_MOE_ROUTER') or 'default'
d['infinilm_moe_router_engine'] = os.environ.get('INFINILM_MOE_ROUTER_ENGINE', '')
d['infinilm_e2e_python'] = '$VPY'
d['infinilm_e2e_attn'] = 'default'
d['infinilm_e2e_cache_type'] = 'static'
json.dump(d, open(p, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
" "$out_json"
}

run_perf "$ART/e2e_infini_server_vllm_fused_router_engine_vllm_poc_c1_hi.json" \
  --base-url "http://127.0.0.1:${PORT}/v1" \
  --model "$MODEL_ID" \
  --num-requests 8 \
  --concurrency 1 \
  --max-tokens 4 \
  --prompt "Hi" \
  --warmup-requests 2

run_perf "$ART/e2e_infini_server_vllm_fused_router_engine_vllm_poc_c8.json" \
  --base-url "http://127.0.0.1:${PORT}/v1" \
  --model "$MODEL_ID" \
  --num-requests 8 \
  --concurrency 8 \
  --max-tokens 64 \
  --warmup-requests 1

echo "=== vllm_poc router e2e summary ===" >&2
"$VPY" -c "
import json, os
art = '$ART'
for name in (
  'e2e_infini_server_vllm_fused_router_engine_vllm_poc_c1_hi.json',
  'e2e_infini_server_vllm_fused_router_engine_vllm_poc_c8.json',
):
  p = os.path.join(art, name)
  d = json.load(open(p, encoding='utf-8'))
  print(name, 'rps=', round(d.get('requests_per_second', 0), 3),
        'avg_ttft_s=', round(d.get('avg_ttft_s', 0), 4),
        'avg_decode_ms_per_chunk=', round(d.get('avg_decode_ms_per_chunk', 0) or 0, 2),
        'avg_latency_s=', round(d.get('avg_latency_s', 0), 3))
"

echo "[e2e-vllm-poc] DONE" >&2
