#!/usr/bin/env bash
# Compare test_perf decode proxy vs bench_balanced smoke ITL (c=1 Hi),
# then test_perf c=8 vs c=1 on same InfiniLM server (random PROMPTS).
# Run inside minicpm5-moe. Do not use `pkill -f infinilm...` from an inline
# `bash -lc '...pkill...'` — it can match the parent shell argv.
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
REPO="${REPO:-/home/zenghua/workspace/minicpm5-moe-support}"
MODEL="${MODEL:-/data-aisoft/zenghua/models/minicpm5.16a3.v0314}"
export REPO
export MODEL
ART="$REPO/InfiniLM/examples/bench_artifacts"

export PYTHONPATH="$REPO/InfiniLM/python:$REPO/InfiniCore/python:${PYTHONPATH:-}"
TORCH_LIB="$(python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), \"lib\"))")"
unset LD_LIBRARY_PATH
unset LD_PRELOAD
export LD_LIBRARY_PATH="/root/.infini/lib:$TORCH_LIB:/usr/local/lib/python3.12/dist-packages:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu"

free_port() {
  local port="$1"
  python3 - <<PY || true
import os, signal
port = int("$port")
try:
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", port))
    s.close()
except OSError:
    os.system(f"fuser -k {port}/tcp 2>/dev/null || true")
PY
  sleep 2
}

start_infini() {
  free_port 8001
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
    --port 8001 \
    --host 0.0.0.0 >/tmp/infini_compare.log 2>&1 &
  INF_PID=$!
  echo "$INF_PID" >/tmp/infini_compare.pid
  for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:8001/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$INF_PID" 2>/dev/null; then
      echo "InfiniLM server died during startup" >&2
      tail -80 /tmp/infini_compare.log >&2 || true
      exit 1
    fi
    sleep 5
  done
  echo "Timeout waiting for InfiniLM on :8001" >&2
  tail -80 /tmp/infini_compare.log >&2 || true
  exit 1
}

stop_infini() {
  if [[ -f /tmp/infini_compare.pid ]]; then
    kill "$(cat /tmp/infini_compare.pid)" 2>/dev/null || true
    rm -f /tmp/infini_compare.pid
  fi
  sleep 2
}

trap stop_infini EXIT

start_infini
python3 -m pip install -q "openai>=1.0"

echo "=== test_perf c=1 Hi (compare to smoke ITL) ==="
python3 "$REPO/InfiniLM/scripts/test_perf.py" \
  --base-url "http://127.0.0.1:8001/v1" \
  --model "minicpm5.16a3.v0314" \
  --num-requests 16 \
  --concurrency 1 \
  --max-tokens 4 \
  --prompt "Hi" \
  --warmup-requests 2 \
  --json-out "$ART/compare_c1_e2e_hi.json"

echo "=== test_perf c=8 random PROMPTS (concurrency stress) ==="
python3 "$REPO/InfiniLM/scripts/test_perf.py" \
  --base-url "http://127.0.0.1:8001/v1" \
  --model "minicpm5.16a3.v0314" \
  --num-requests 32 \
  --concurrency 8 \
  --max-tokens 128 \
  --warmup-requests 2 \
  --json-out "$ART/compare_concurrency_c8.json"

echo "=== test_perf c=1 random PROMPTS (same counts as c8) ==="
python3 "$REPO/InfiniLM/scripts/test_perf.py" \
  --base-url "http://127.0.0.1:8001/v1" \
  --model "minicpm5.16a3.v0314" \
  --num-requests 32 \
  --concurrency 1 \
  --max-tokens 128 \
  --warmup-requests 2 \
  --json-out "$ART/compare_concurrency_c1.json"

stop_infini
trap - EXIT

echo "=== bench_balanced in-proc smoke (Hi, 7 tok prompt, 4 new) ==="
cd "$REPO/InfiniLM/examples"
INFINILM_SUPPRESS_BENCH_PRINTS=1 python3 bench_balanced.py --nvidia \
  --model-path "$MODEL" \
  --attn flash-attn --enable-paged-attn --paged-kv-block-size 256 \
  --prompt "Hi" --prompt-tokens 7 --max-new-tokens 4 \
  --warmup-steps 3 --timing-discard-runs 1 --runs 1 \
  --json-out "$ART/compare_c1_smoke_hi.json"

python3 - <<'PY'
import json, os

REPO = os.environ["REPO"]
base = os.path.join(REPO, "InfiniLM/examples/bench_artifacts")

def load(name):
    with open(os.path.join(base, name)) as f:
        return json.load(f)

e2e = load("compare_c1_e2e_hi.json")
sm = load("compare_c1_smoke_hi.json")
e2e_d = float(e2e["avg_decode_ms_per_chunk"])
sm_d = float(sm["avg_decode_itl_ms"])
c8 = load("compare_concurrency_c8.json")
c1 = load("compare_concurrency_c1.json")

summary_c1 = {
    "e2e_avg_decode_ms_per_chunk": e2e_d,
    "smoke_avg_decode_itl_ms": sm_d,
    "abs_diff_ms": abs(e2e_d - sm_d),
    "ratio_e2e_over_smoke": e2e_d / sm_d if sm_d else None,
    "note": "e2e: test_perf OpenAI stream c=1 Hi max_tokens=4. smoke: bench_balanced Hi prompt_tokens=7 max_new_tokens=4.",
}

summary_conc = {
    "c8": {
        "concurrency": c8["concurrency"],
        "avg_ms_per_token": c8["avg_ms_per_token"],
        "avg_decode_ms_per_chunk": c8["avg_decode_ms_per_chunk"],
        "avg_latency_s": c8["avg_latency_s"],
        "requests_per_second": c8["requests_per_second"],
    },
    "c1": {
        "concurrency": c1["concurrency"],
        "avg_ms_per_token": c1["avg_ms_per_token"],
        "avg_decode_ms_per_chunk": c1["avg_decode_ms_per_chunk"],
        "avg_latency_s": c1["avg_latency_s"],
        "requests_per_second": c1["requests_per_second"],
    },
    "avg_ms_per_token_ratio_c8_over_c1": (
        (float(c8["avg_ms_per_token"]) / float(c1["avg_ms_per_token"]))
        if c1.get("avg_ms_per_token") and c8.get("avg_ms_per_token")
        else None
    ),
    "interpretation": (
        "avg_ms_per_token / avg_ms_per_stream_chunk is elapsed_wall / streamed_text_chunks per request, averaged. "
        "At c=8, inter-chunk gaps include other requests' GPU steps in the same AsyncLLMEngine batch, "
        "so it inflates vs c=1 for the same per-request chunk count regime."
    ),
}

with open(os.path.join(base, "compare_c1_decode_summary.json"), "w") as f:
    json.dump(summary_c1, f, indent=2)
with open(os.path.join(base, "compare_concurrency_summary.json"), "w") as f:
    json.dump(summary_conc, f, indent=2)

print(json.dumps({"c1_hi_decode_compare": summary_c1, "concurrency": summary_conc}, indent=2))
PY

echo "DONE_COMPARE_CONCURRENCY"
