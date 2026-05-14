#!/usr/bin/env bash
# InfiniLM OpenAI server: MoE backend benefit (baseline vs vllm_fused) using test_perf.py.
# Run inside minicpm5-moe (or host with same paths). Two full server boots — MoE backend is fixed at load.
#
# Pick GPU **inside** `docker exec` (do not rely on `CUDA_VISIBLE_DEVICES=N` only on the host):
#
#   docker exec minicpm5-moe bash -lc 'export CUDA_VISIBLE_DEVICES=1; bash /home/zenghua/workspace/minicpm5-moe-support/InfiniLM/examples/run_e2e_server_moe_benefit.sh'
#
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
REPO="${REPO:-/home/zenghua/workspace/minicpm5-moe-support}"
MODEL="${MODEL:-/data-aisoft/zenghua/models/minicpm5.16a3.v0314}"
PORT="${PORT:-8016}"
MODEL_ID="$(basename "$MODEL")"

export PYTHONPATH="$REPO/InfiniLM/python:$REPO/InfiniCore/python:${PYTHONPATH:-}"
TORCH_LIB="$(python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), \"lib\"))")"
unset LD_LIBRARY_PATH
unset LD_PRELOAD
export LD_LIBRARY_PATH="/root/.infini/lib:$TORCH_LIB:/usr/local/lib/python3.12/dist-packages:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu"

ART="$REPO/InfiniLM/examples/bench_artifacts"
mkdir -p "$ART"
PIDFILE="/tmp/infini_moe_e2e_${PORT}.pid"
LOGFILE="/tmp/infini_moe_e2e_${PORT}.log"

python3 -m pip install -q "openai>=1.0" 2>/dev/null || true

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

start_server() {
  local moe_mode="$1"
  cleanup
  free_port
  sleep 1

  case "$moe_mode" in
    baseline) export INFINILM_FORCE_MOE_BACKEND=baseline ;;
    vllm_fused) export INFINILM_FORCE_MOE_BACKEND=vllm_fused ;;
    *) echo "unknown moe_mode=$moe_mode" >&2; exit 2 ;;
  esac

  echo "[e2e] starting server port=$PORT INFINILM_FORCE_MOE_BACKEND=${INFINILM_FORCE_MOE_BACKEND:-}" >&2
  cd "$REPO/InfiniLM/python"
  python3 -m infinilm.server.inference_server --nvidia \
    --model_path "$MODEL" \
    --dtype bfloat16 \
    --attn flash-attn \
    --cache_type paged \
    --num_blocks 256 \
    --block_size 256 \
    --max_batch_size 8 \
    --max_tokens 256 \
    --port "$PORT" \
    --host 127.0.0.1 >"$LOGFILE" 2>&1 &
  echo $! >"$PIDFILE"

  for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
      echo "[e2e] server ready" >&2
      return 0
    fi
    if ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "[e2e] server died during startup" >&2
      tail -80 "$LOGFILE" >&2 || true
      exit 1
    fi
    sleep 2
  done
  echo "[e2e] timeout waiting for /v1/models" >&2
  tail -80 "$LOGFILE" >&2 || true
  exit 1
}

tag_json() {
  local path="$1"
  local backend="$2"
  python3 -c "import json,sys; p=sys.argv[1]; b=sys.argv[2]; d=json.load(open(p,encoding='utf-8')); d['infinilm_moe_backend']=b; json.dump(d,open(p,'w',encoding='utf-8'),indent=2,ensure_ascii=False)" "$path" "$backend"
}

run_perf() {
  local out_json="$1"
  shift
  echo "[e2e] test_perf -> $out_json $*" >&2
  python3 "$REPO/InfiniLM/scripts/test_perf.py" "$@" --json-out "$out_json"
  local be="${INFINILM_FORCE_MOE_BACKEND:-}"
  tag_json "$out_json" "$be"
}

# --- baseline server ---
start_server baseline
run_perf "$ART/e2e_infini_server_baseline_c1_hi.json" \
  --base-url "http://127.0.0.1:${PORT}/v1" \
  --model "$MODEL_ID" \
  --num-requests 8 \
  --concurrency 1 \
  --max-tokens 4 \
  --prompt "Hi" \
  --warmup-requests 2

run_perf "$ART/e2e_infini_server_baseline_c8.json" \
  --base-url "http://127.0.0.1:${PORT}/v1" \
  --model "$MODEL_ID" \
  --num-requests 8 \
  --concurrency 8 \
  --max-tokens 64 \
  --warmup-requests 1

cleanup
sleep 3

# --- vllm_fused server ---
start_server vllm_fused
run_perf "$ART/e2e_infini_server_vllm_fused_c1_hi.json" \
  --base-url "http://127.0.0.1:${PORT}/v1" \
  --model "$MODEL_ID" \
  --num-requests 8 \
  --concurrency 1 \
  --max-tokens 4 \
  --prompt "Hi" \
  --warmup-requests 2

run_perf "$ART/e2e_infini_server_vllm_fused_c8.json" \
  --base-url "http://127.0.0.1:${PORT}/v1" \
  --model "$MODEL_ID" \
  --num-requests 8 \
  --concurrency 8 \
  --max-tokens 64 \
  --warmup-requests 1

echo "=== E2E MoE benefit summary (see JSON under $ART) ===" >&2
python3 -c "
import json, os
art = '$ART'
rows = [
  'e2e_infini_server_baseline_c1_hi.json',
  'e2e_infini_server_baseline_c8.json',
  'e2e_infini_server_vllm_fused_c1_hi.json',
  'e2e_infini_server_vllm_fused_c8.json',
]
for name in rows:
  p = os.path.join(art, name)
  d = json.load(open(p, encoding='utf-8'))
  print(name, 'backend=', d.get('infinilm_moe_backend'), 'rps=', round(d.get('requests_per_second', 0), 3),
        'avg_ttft_s=', round(d.get('avg_ttft_s', 0), 4),
        'avg_decode_ms_per_chunk=', round(d.get('avg_decode_ms_per_chunk', 0) or 0, 2),
        'avg_latency_s=', round(d.get('avg_latency_s', 0), 3))
"

echo "[e2e] DONE" >&2
