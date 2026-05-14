#!/usr/bin/env bash
# Two-stack OpenAI e2e: **vendor** (system python3, flash-attn + paged KV, INFINILM_MOE_FUSED_STACK=vendor)
# vs **upstream** (.venv-vllm, default attn + static KV, INFINILM_MOE_FUSED_STACK=upstream).
#
# MoE router + fused experts differ by design; attention / KV / torch builds differ — do not claim parity.
# Writes the same JSON filenames as `run_e2e_server_moe_benefit.sh` (vendor) and
# `run_e2e_server_moe_vllm_poc_router.sh` (upstream), plus `bench_artifacts/e2e_moe_fused_stack_compare_summary.json`.
#
#   docker exec minicpm5-moe bash -lc 'export CUDA_VISIBLE_DEVICES=1; bash /home/zenghua/workspace/minicpm5-moe-support/InfiniLM/examples/run_e2e_moe_fused_stack_compare.sh'
#
# Env:
#   SKIP_VENDOR_E2E=1   — only run upstream leg (reuse existing vendor JSONs if present).
#   SKIP_UPSTREAM_E2E=1 — only run vendor leg.
#   PORT_VENDOR (default 8016), PORT_UPSTREAM (default 8017; exported for nested poc script).
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

export PYTHONPATH="$REPO/InfiniLM/python:$REPO/InfiniCore/python:${PYTHONPATH:-}"

ART="$REPO/InfiniLM/examples/bench_artifacts"
mkdir -p "$ART"

vendor_pidfile="/tmp/infini_moe_stack_cmp_vendor_${PORT_VENDOR}.pid"
vendor_log="/tmp/infini_moe_stack_cmp_vendor_${PORT_VENDOR}.log"

cleanup_vendor() {
  if [[ -f "$vendor_pidfile" ]]; then
    kill "$(cat "$vendor_pidfile")" 2>/dev/null || true
    rm -f "$vendor_pidfile"
  fi
}
trap cleanup_vendor EXIT

free_port_vendor() {
  python3 - <<PY || true
import os, socket
port = int("$PORT_VENDOR")
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.bind(("127.0.0.1", port))
    s.close()
except OSError:
    os.system(f"fuser -k {port}/tcp 2>/dev/null || true")
PY
  sleep 2
}

start_vendor_server() {
  cleanup_vendor
  free_port_vendor
  sleep 1

  unset INFINILM_MOE_ROUTER
  unset INFINILM_MOE_ROUTER_ENGINE
  export INFINILM_FORCE_MOE_BACKEND=vllm_fused
  export INFINILM_MOE_FUSED_STACK=vendor

  TORCH_LIB="$(python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), \"lib\"))")"
  unset LD_LIBRARY_PATH
  unset LD_PRELOAD
  export LD_LIBRARY_PATH="/root/.infini/lib:$TORCH_LIB:/usr/local/lib/python3.12/dist-packages:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu"

  echo "[stack-compare] vendor server port=$PORT_VENDOR python=$(command -v python3) stack=vendor attn=flash-attn cache=paged" >&2
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
      echo "[stack-compare] vendor server ready" >&2
      return 0
    fi
    if ! kill -0 "$(cat "$vendor_pidfile")" 2>/dev/null; then
      echo "[stack-compare] vendor server died during startup" >&2
      tail -80 "$vendor_log" >&2 || true
      exit 1
    fi
    sleep 2
  done
  echo "[stack-compare] vendor server timeout" >&2
  tail -80 "$vendor_log" >&2 || true
  exit 1
}

tag_vendor_json() {
  local path="$1"
  python3 -c "
import json, sys
p = sys.argv[1]
d = json.load(open(p, encoding='utf-8'))
d['infinilm_moe_backend'] = 'vllm_fused'
d['infinilm_moe_fused_stack'] = 'vendor'
d['infinilm_e2e_python'] = '$(command -v python3)'
d['infinilm_e2e_attn'] = 'flash-attn'
d['infinilm_e2e_cache_type'] = 'paged'
json.dump(d, open(p, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
" "$path"
}

run_vendor_perf() {
  local out_json="$1"
  shift
  echo "[stack-compare] test_perf (vendor) -> $out_json $*" >&2
  python3 "$REPO/InfiniLM/scripts/test_perf.py" "$@" --json-out "$out_json"
  tag_vendor_json "$out_json"
}

run_vendor_leg() {
  python3 -m pip install -q "openai>=1.0" 2>/dev/null || true
  start_vendor_server
  run_vendor_perf "$ART/e2e_infini_server_vllm_fused_c1_hi.json" \
    --base-url "http://127.0.0.1:${PORT_VENDOR}/v1" \
    --model "$MODEL_ID" \
    --num-requests 8 \
    --concurrency 1 \
    --max-tokens 4 \
    --prompt "Hi" \
    --warmup-requests 2
  run_vendor_perf "$ART/e2e_infini_server_vllm_fused_c8.json" \
    --base-url "http://127.0.0.1:${PORT_VENDOR}/v1" \
    --model "$MODEL_ID" \
    --num-requests 8 \
    --concurrency 8 \
    --max-tokens 64 \
    --warmup-requests 1
  cleanup_vendor
  sleep 2
}

if [[ "${SKIP_VENDOR_E2E:-0}" != "1" ]]; then
  run_vendor_leg
else
  echo "[stack-compare] SKIP_VENDOR_E2E=1 — not booting vendor server" >&2
fi

if [[ "${SKIP_UPSTREAM_E2E:-0}" != "1" ]]; then
  if [[ ! -x "$VPY" ]]; then
    echo "[stack-compare] error: upstream venv missing: $VPY (set VPY or create .venv-vllm)" >&2
    exit 1
  fi
  export PORT="$PORT_UPSTREAM"
  bash "$REPO/InfiniLM/examples/run_e2e_server_moe_vllm_poc_router.sh"
else
  echo "[stack-compare] SKIP_UPSTREAM_E2E=1 — not booting upstream server" >&2
fi

python3 - <<PY
import json, os, math

REPO = os.environ["REPO"]
ART = os.path.join(REPO, "InfiniLM", "examples", "bench_artifacts")

pairs = [
    ("c1_hi", "e2e_infini_server_vllm_fused_c1_hi.json", "e2e_infini_server_vllm_fused_upstream_c1_hi.json"),
    ("c8", "e2e_infini_server_vllm_fused_c8.json", "e2e_infini_server_vllm_fused_upstream_c8.json"),
]

rows = []
for label, vname, uname in pairs:
    vp = os.path.join(ART, vname)
    up = os.path.join(ART, uname)
    if not os.path.isfile(vp) or not os.path.isfile(up):
        print(f"[stack-compare] warn: missing {vname!r} or {uname!r}; skip row {label}", flush=True)
        continue
    v = json.load(open(vp, encoding="utf-8"))
    u = json.load(open(up, encoding="utf-8"))

    def dget(d, k):
        x = d.get(k)
        return float(x) if x is not None and not (isinstance(x, float) and math.isnan(x)) else None

    keys = (
        "requests_per_second",
        "avg_ttft_s",
        "avg_decode_ms_per_chunk",
        "avg_latency_s",
    )
    entry = {"label": label, "vendor_json": vname, "upstream_json": uname}
    for k in keys:
        vv, uv = dget(v, k), dget(u, k)
        entry[f"vendor_{k}"] = vv
        entry[f"upstream_{k}"] = uv
        if vv is not None and uv is not None:
            entry[f"delta_{k}_upstream_minus_vendor"] = uv - vv
        else:
            entry[f"delta_{k}_upstream_minus_vendor"] = None
    rows.append(entry)

summary = {
    "note": "Deltas are upstream − vendor. Stacks differ in attn/KV/torch; interpret as exploratory.",
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    "rows": rows,
}

outp = os.path.join(ART, "e2e_moe_fused_stack_compare_summary.json")
json.dump(summary, open(outp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print(f"[stack-compare] wrote {outp}", flush=True)

print("")
print("=== e2e vendor vs upstream (upstream − vendor) ===")
for r in rows:
    print(f"--- {r['label']} ---")
    for k in ("requests_per_second", "avg_ttft_s", "avg_decode_ms_per_chunk", "avg_latency_s"):
        dv = r.get(f"delta_{k}_upstream_minus_vendor")
        if dv is None:
            continue
        print(f"  Δ {k}: {dv:+.6g}")
PY

echo "[stack-compare] DONE" >&2
