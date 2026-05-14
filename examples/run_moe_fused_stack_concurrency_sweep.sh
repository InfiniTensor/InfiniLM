#!/usr/bin/env bash
# Golden matrix: same test_perf flags, sweep --concurrency for vendor vs upstream stacks.
#
# Vendor: system python3, flash-attn, paged KV, INFINILM_MOE_FUSED_STACK=vendor (port PORT_VENDOR).
# Upstream: .venv-vllm, default attn, static KV, INFINILM_MOE_FUSED_STACK=upstream (port PORT_UPSTREAM).
#
# JSON naming:
#   e2e_moe_stack_vendor_concurrency_c{N}_hi.json
#   e2e_moe_stack_upstream_concurrency_c{N}_hi.json
#
# Mixed-stack caveat: attn/KV/torch differ between legs — interpret cross-stack deltas as exploratory
# (same as run_e2e_moe_fused_stack_compare.sh). Same-stack c curves isolate scheduler + client concurrency.
#
#   docker exec minicpm5-moe bash -lc 'export CUDA_VISIBLE_DEVICES=1; bash /home/zenghua/workspace/minicpm5-moe-support/InfiniLM/examples/run_moe_fused_stack_concurrency_sweep.sh'
#
# Env:
#   CONCURRENCIES="1 2 4 8"   — default
#   SKIP_VENDOR=1 | SKIP_UPSTREAM=1
#   PORT_VENDOR (8016), PORT_UPSTREAM (8017)
#   NUM_REQUESTS (16), MAX_TOKENS (4), WARMUP (2), PROMPT (Hi)
#
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
REPO="${REPO:-/home/zenghua/workspace/minicpm5-moe-support}"
export REPO
MODEL="${MODEL:-/data-aisoft/zenghua/models/minicpm5.16a3.v0314}"
PORT_VENDOR="${PORT_VENDOR:-8016}"
PORT_UPSTREAM="${PORT_UPSTREAM:-8017}"
MODEL_ID="$(basename "$MODEL")"
VPY="${VPY:-$REPO/.venv-vllm/bin/python}"

CONCURRENCIES="${CONCURRENCIES:-1 2 4 8}"
export CONCURRENCIES
NUM_REQUESTS="${NUM_REQUESTS:-16}"
MAX_TOKENS="${MAX_TOKENS:-4}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-2}"
FIXED_PROMPT="${FIXED_PROMPT:-Hi}"
export NUM_REQUESTS MAX_TOKENS WARMUP_REQUESTS FIXED_PROMPT

export PYTHONPATH="$REPO/InfiniLM/python:$REPO/InfiniCore/python:${PYTHONPATH:-}"
ART="$REPO/InfiniLM/examples/bench_artifacts"
mkdir -p "$ART"

vendor_pidfile="/tmp/infini_moe_conc_vendor_${PORT_VENDOR}.pid"
vendor_log="/tmp/infini_moe_conc_vendor_${PORT_VENDOR}.log"
upstream_pidfile="/tmp/infini_moe_conc_upstream_${PORT_UPSTREAM}.pid"
upstream_log="/tmp/infini_moe_conc_upstream_${PORT_UPSTREAM}.log"

cleanup_all() {
  if [[ -f "$vendor_pidfile" ]]; then
    kill "$(cat "$vendor_pidfile")" 2>/dev/null || true
    rm -f "$vendor_pidfile"
  fi
  if [[ -f "$upstream_pidfile" ]]; then
    kill "$(cat "$upstream_pidfile")" 2>/dev/null || true
    rm -f "$upstream_pidfile"
  fi
}
trap cleanup_all EXIT

free_port() {
  local port="$1"
  python3 - <<PY || true
import os, socket
port = int("$port")
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.bind(("127.0.0.1", port))
    s.close()
except OSError:
    os.system(f"fuser -k {port}/tcp 2>/dev/null || true")
PY
  sleep 2
}

start_vendor() {
  cleanup_all
  free_port "$PORT_VENDOR"
  sleep 1
  unset INFINILM_MOE_ROUTER
  unset INFINILM_MOE_ROUTER_ENGINE
  export INFINILM_FORCE_MOE_BACKEND=vllm_fused
  export INFINILM_MOE_FUSED_STACK=vendor
  TORCH_LIB="$(python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), \"lib\"))")"
  unset LD_LIBRARY_PATH
  unset LD_PRELOAD
  export LD_LIBRARY_PATH="/root/.infini/lib:$TORCH_LIB:/usr/local/lib/python3.12/dist-packages:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu"
  echo "[conc-sweep] vendor server port=$PORT_VENDOR" >&2
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
    --port "$PORT_VENDOR" \
    --host 127.0.0.1 >"$vendor_log" 2>&1 &
  echo $! >"$vendor_pidfile"
  for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:${PORT_VENDOR}/v1/models" >/dev/null 2>&1; then
      echo "[conc-sweep] vendor ready" >&2
      return 0
    fi
    if ! kill -0 "$(cat "$vendor_pidfile")" 2>/dev/null; then
      tail -80 "$vendor_log" >&2 || true
      exit 1
    fi
    sleep 2
  done
  exit 1
}

start_upstream() {
  cleanup_all
  free_port "$PORT_UPSTREAM"
  sleep 1
  export INFINILM_FORCE_MOE_BACKEND=vllm_fused
  export INFINILM_MOE_FUSED_STACK=upstream
  TORCH_LIB="$("$VPY" -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), \"lib\"))")"
  unset LD_LIBRARY_PATH
  unset LD_PRELOAD
  export LD_LIBRARY_PATH="/root/.infini/lib:$TORCH_LIB:/usr/local/lib/python3.12/dist-packages:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu"
  echo "[conc-sweep] upstream server port=$PORT_UPSTREAM" >&2
  cd "$REPO/InfiniLM/python"
  "$VPY" -m infinilm.server.inference_server --nvidia \
    --model_path "$MODEL" \
    --dtype bfloat16 \
    --attn default \
    --cache_type static \
    --num_blocks 256 \
    --block_size 256 \
    --max_batch_size 8 \
    --max_tokens 256 \
    --port "$PORT_UPSTREAM" \
    --host 127.0.0.1 >"$upstream_log" 2>&1 &
  echo $! >"$upstream_pidfile"
  for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:${PORT_UPSTREAM}/v1/models" >/dev/null 2>&1; then
      echo "[conc-sweep] upstream ready" >&2
      return 0
    fi
    if ! kill -0 "$(cat "$upstream_pidfile")" 2>/dev/null; then
      tail -120 "$upstream_log" >&2 || true
      exit 1
    fi
    sleep 2
  done
  exit 1
}

tag_vendor_json() {
  local path="$1" c="$2"
  python3 -c "
import json, sys
p, c = sys.argv[1], int(sys.argv[2])
d = json.load(open(p, encoding='utf-8'))
d['infinilm_moe_backend'] = 'vllm_fused'
d['infinilm_moe_fused_stack'] = 'vendor'
d['infinilm_e2e_python'] = '$(command -v python3)'
d['infinilm_e2e_attn'] = 'flash-attn'
d['infinilm_e2e_cache_type'] = 'paged'
d['infinilm_concurrency_sweep_c'] = c
json.dump(d, open(p, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
" "$path" "$c"
}

tag_upstream_json() {
  local path="$1" c="$2"
  "$VPY" -c "
import json, sys
p, c = sys.argv[1], int(sys.argv[2])
d = json.load(open(p, encoding='utf-8'))
d['infinilm_moe_backend'] = 'vllm_fused'
d['infinilm_moe_fused_stack'] = 'upstream'
d['infinilm_e2e_python'] = '$VPY'
d['infinilm_e2e_attn'] = 'default'
d['infinilm_e2e_cache_type'] = 'static'
d['infinilm_concurrency_sweep_c'] = c
json.dump(d, open(p, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
" "$path" "$c"
}

python3 -m pip install -q "openai>=1.0" 2>/dev/null || true
"$VPY" -m pip install -q "openai>=1.0" janus xxhash fastapi uvicorn pydantic 2>/dev/null || true

rows_vendor=()
rows_upstream=()

if [[ "${SKIP_VENDOR:-0}" != "1" ]]; then
  start_vendor
  for c in $CONCURRENCIES; do
    out="$ART/e2e_moe_stack_vendor_concurrency_c${c}_hi.json"
    echo "[conc-sweep] vendor test_perf c=$c -> $out" >&2
    python3 "$REPO/InfiniLM/scripts/test_perf.py" \
      --base-url "http://127.0.0.1:${PORT_VENDOR}/v1" \
      --model "$MODEL_ID" \
      --num-requests "$NUM_REQUESTS" \
      --concurrency "$c" \
      --max-tokens "$MAX_TOKENS" \
      --prompt "$FIXED_PROMPT" \
      --warmup-requests "$WARMUP_REQUESTS" \
      --json-out "$out"
    tag_vendor_json "$out" "$c"
    rows_vendor+=("$out")
  done
  cleanup_all
  sleep 2
fi

if [[ "${SKIP_UPSTREAM:-0}" != "1" ]]; then
  if [[ ! -x "$VPY" ]]; then
    echo "[conc-sweep] error: missing $VPY" >&2
    exit 1
  fi
  start_upstream
  for c in $CONCURRENCIES; do
    out="$ART/e2e_moe_stack_upstream_concurrency_c${c}_hi.json"
    echo "[conc-sweep] upstream test_perf c=$c -> $out" >&2
    "$VPY" "$REPO/InfiniLM/scripts/test_perf.py" \
      --base-url "http://127.0.0.1:${PORT_UPSTREAM}/v1" \
      --model "$MODEL_ID" \
      --num-requests "$NUM_REQUESTS" \
      --concurrency "$c" \
      --max-tokens "$MAX_TOKENS" \
      --prompt "$FIXED_PROMPT" \
      --warmup-requests "$WARMUP_REQUESTS" \
      --json-out "$out"
    tag_upstream_json "$out" "$c"
    rows_upstream+=("$out")
  done
  cleanup_all
fi

python3 - <<PY
import json, os, math

REPO = os.environ["REPO"]
ART = os.path.join(REPO, "InfiniLM", "examples", "bench_artifacts")

def loadj(name):
    p = os.path.join(ART, name)
    if not os.path.isfile(p):
        return None
    return json.load(open(p, encoding="utf-8"))

def dget(d, k):
    if d is None:
        return None
    x = d.get(k)
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    return float(x)

conc_list = [int(x) for x in os.environ.get("CONCURRENCIES", "1 2 4 8").split()]
rows = []
for c in conc_list:
    vname = f"e2e_moe_stack_vendor_concurrency_c{c}_hi.json"
    uname = f"e2e_moe_stack_upstream_concurrency_c{c}_hi.json"
    v, u = loadj(vname), loadj(uname)
    entry = {
        "concurrency": c,
        "vendor_json": vname if v else None,
        "upstream_json": uname if u else None,
    }
    for k in ("requests_per_second", "avg_ttft_s", "avg_decode_ms_per_chunk", "avg_latency_s"):
        vv, uv = dget(v, k), dget(u, k)
        entry[f"vendor_{k}"] = vv
        entry[f"upstream_{k}"] = uv
        if vv is not None and uv is not None:
            entry[f"delta_{k}_upstream_minus_vendor"] = uv - vv
        else:
            entry[f"delta_{k}_upstream_minus_vendor"] = None
    rows.append(entry)

summary = {
    "note": "Golden matrix: same prompt/max_tokens/num_requests; only concurrency varies per row. "
    "Cross-stack deltas mix attn/KV/torch — see MOE_VENDOR_UPSTREAM_CONCURRENCY_MEMO.md.",
    "golden": {
        "num_requests": int(os.environ.get("NUM_REQUESTS", "16")),
        "max_tokens": int(os.environ.get("MAX_TOKENS", "4")),
        "prompt": os.environ.get("FIXED_PROMPT", "Hi"),
        "warmup_requests": int(os.environ.get("WARMUP_REQUESTS", "2")),
    },
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    "rows": rows,
}
outp = os.path.join(ART, "e2e_moe_fused_stack_concurrency_summary.json")
json.dump(summary, open(outp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print(f"[conc-sweep] wrote {outp}")
for r in rows:
    c = r["concurrency"]
    print(f"--- c={c} ---")
    for k in ("requests_per_second", "avg_ttft_s", "avg_decode_ms_per_chunk", "avg_latency_s"):
        dv = r.get(f"delta_{k}_upstream_minus_vendor")
        if dv is None:
            continue
        print(f"  Δ {k} (upstream−vendor): {dv:+.6g}")
PY

echo "[conc-sweep] DONE" >&2
