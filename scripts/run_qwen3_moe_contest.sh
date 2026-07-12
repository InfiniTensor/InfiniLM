#!/usr/bin/env bash
# Run the T2-1-3 Qwen3-MoE correctness or serving benchmark profiles.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

usage() {
    cat <<'EOF'
Usage: scripts/run_qwen3_moe_contest.sh PROFILE

Profiles:
  check       Run the focused policy tests without starting a server.
  smoke       Start a small server and issue one real chat request.
  c1-short    Benchmark concurrency=1, input=32, output=256.
  c64-short   Benchmark concurrency=64, input=32, output=256.
  c64-long    Benchmark concurrency=64, input=256, output=1024.

Environment:
  MODEL_DIR        Qwen3-30B-A3B-Thinking-2507 directory (required except for check)
  INFINICORE_DIR   Optional InfiniCore checkout; omit when installed in Python
  INFINI_ROOT      Installed InfiniCore prefix (default: $HOME/.infini)
  PYTHON           Server Python executable (default: python3)
  BENCH_PYTHON     Python with vLLM 0.9.2 client modules (default: PYTHON)
  RESULT_DIR       Log/result directory (default: current-directory benchmark_results/)
  CUDA_VISIBLE_DEVICES  Four TP devices (default: 0,1,2,3)
  PORT             Local service port (default: 8102)
  BUILD=1          Rebuild/install InfiniLM before running the profile
  DRY_RUN=1        Print the resolved commands without executing them
  SHOW_COMMANDS=1  Print full commands during a real run

Examples:
  MODEL_DIR=/path/to/Qwen3-30B-A3B-Thinking-2507 \
    scripts/run_qwen3_moe_contest.sh smoke

  MODEL_DIR=/path/to/Qwen3-30B-A3B-Thinking-2507 \
    BENCH_PYTHON=/path/to/vllm-client/bin/python \
    scripts/run_qwen3_moe_contest.sh c64-long
EOF
}

PROFILE=${1:-}
if [ -z "$PROFILE" ] || [ "$PROFILE" = "-h" ] || [ "$PROFILE" = "--help" ]; then
    usage
    exit 0
fi

MODEL_DIR=${MODEL_DIR:-}
INFINICORE_DIR=${INFINICORE_DIR:-}
INFINI_ROOT=${INFINI_ROOT:-${HOME:-$REPO_DIR}/.infini}
PYTHON=${PYTHON:-python3}
BENCH_PYTHON=${BENCH_PYTHON:-$PYTHON}
RESULT_DIR=${RESULT_DIR:-benchmark_results/qwen3_moe}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8102}
TP=${TP:-4}
BUILD=${BUILD:-0}
DRY_RUN=${DRY_RUN:-0}
SHOW_COMMANDS=${SHOW_COMMANDS:-0}
SERVER_START_TIMEOUT=${SERVER_START_TIMEOUT:-1800}

MAX_BATCH_SIZE=1
NUM_BLOCKS=18
NUM_PROMPTS=1
MAX_CONCURRENCY=1
INPUT_LEN=32
OUTPUT_LEN=128
RUN_BENCH=0

case "$PROFILE" in
    check)
        ;;
    smoke)
        ;;
    c1-short)
        OUTPUT_LEN=256
        RUN_BENCH=1
        ;;
    c64-short)
        MAX_BATCH_SIZE=64
        NUM_BLOCKS=128
        NUM_PROMPTS=64
        MAX_CONCURRENCY=64
        OUTPUT_LEN=256
        RUN_BENCH=1
        ;;
    c64-long)
        MAX_BATCH_SIZE=64
        NUM_BLOCKS=120
        NUM_PROMPTS=64
        MAX_CONCURRENCY=64
        INPUT_LEN=256
        OUTPUT_LEN=1024
        RUN_BENCH=1
        ;;
    *)
        echo "Unknown profile: $PROFILE" >&2
        usage >&2
        exit 2
        ;;
esac

if [ "$PROFILE" != "check" ] && [ -z "$MODEL_DIR" ]; then
    echo "MODEL_DIR is required for profile $PROFILE" >&2
    exit 2
fi

if [ -n "$MODEL_DIR" ]; then
    MODEL_NAME=${MODEL_NAME:-$(basename "$MODEL_DIR")}
else
    MODEL_NAME=${MODEL_NAME:-Qwen3-30B-A3B-Thinking-2507}
fi
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_NAME="${PROFILE}-${TIMESTAMP}"
SERVER_LOG="$RESULT_DIR/${RUN_NAME}-server.log"
CLIENT_LOG="$RESULT_DIR/${RUN_NAME}-client.log"
RESULT_FILE="$RUN_NAME.json"

export INFINI_ROOT
export CUDA_VISIBLE_DEVICES
export LD_LIBRARY_PATH="$INFINI_ROOT/lib:${LD_LIBRARY_PATH:-}"
if [ -n "$INFINICORE_DIR" ]; then
    export PYTHONPATH="$INFINICORE_DIR/python:$REPO_DIR/python:${PYTHONPATH:-}"
else
    export PYTHONPATH="$REPO_DIR/python:${PYTHONPATH:-}"
fi
if [ "$MAX_BATCH_SIZE" -ge 32 ]; then
    export INFINILM_MAX_NUM_BATCHED_TOKENS=4096
else
    unset INFINILM_MAX_NUM_BATCHED_TOKENS || true
fi

SERVER_COMMAND=(
    "$PYTHON" "$REPO_DIR/python/infinilm/server/inference_server.py"
    --device nvidia
    --model "$MODEL_DIR"
    --dtype bfloat16
    --tp "$TP"
    --dp 1
    --ep 1
    --moe-ep-backend disabled
    --host "$HOST"
    --port "$PORT"
    --max-batch-size "$MAX_BATCH_SIZE"
    --max-new-tokens 4096
    --num-blocks "$NUM_BLOCKS"
    --block-size 256
    --temperature 1.0
    --top-p 0.8
    --top-k 1
    --enable-paged-attn
    --attn paged-attn
    --enable-graph
    --moe-backend auto
    --ignore-eos
)

BENCH_COMMAND=(
    "$BENCH_PYTHON" "$SCRIPT_DIR/run_vllm_bench_serve_client.py"
    --backend openai-chat
    --endpoint-type openai-chat
    --model "$MODEL_NAME"
    --tokenizer "$MODEL_DIR"
    --endpoint /v1/chat/completions
    --host "$HOST"
    --port "$PORT"
    --dataset-name random
    --request-rate inf
    --seed 714
    --num-prompts "$NUM_PROMPTS"
    --max-concurrency "$MAX_CONCURRENCY"
    --random-input-len "$INPUT_LEN"
    --random-output-len "$OUTPUT_LEN"
    --temperature 1.0
    --top-p 0.8
    --top-k 1
    --extra-body "{\"max_tokens\": $OUTPUT_LEN}"
    --save-result
    --save-detailed
    --result-dir "$RESULT_DIR"
    --result-filename "$RESULT_FILE"
)

print_command() {
    printf '%q ' "$@"
    printf '\n'
}

echo "Profile: $PROFILE"
if [ -n "$MODEL_DIR" ]; then
    echo "Model: $MODEL_NAME"
fi

if [ "$PROFILE" = "check" ]; then
    CHECK_COMMAND=(
        "$PYTHON" -m pytest -q "$REPO_DIR/test/llm/test_moe_policy.py"
    )
    if [ "$DRY_RUN" = "1" ]; then
        print_command "${CHECK_COMMAND[@]}"
        exit 0
    fi
    "${CHECK_COMMAND[@]}"
    exit 0
fi

if [ "$DRY_RUN" = "1" ]; then
    echo "Server command:"
    print_command "${SERVER_COMMAND[@]}"
    if [ "$RUN_BENCH" = "1" ]; then
        echo "Client command:"
        print_command "${BENCH_COMMAND[@]}"
    fi
    exit 0
fi

if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "Model config not found: $MODEL_DIR/config.json" >&2
    exit 3
fi
if [ -n "$INFINICORE_DIR" ] && [ ! -d "$INFINICORE_DIR/python/infinicore" ]; then
    echo "InfiniCore Python package not found: $INFINICORE_DIR/python/infinicore" >&2
    exit 3
fi
if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required" >&2
    exit 3
fi

mkdir -p "$RESULT_DIR"

if [ "$BUILD" = "1" ]; then
    echo "Building and installing InfiniLM..."
    (
        cd "$REPO_DIR"
        "$PYTHON" -m pip install -e . --no-build-isolation
    )
fi

"$PYTHON" -c \
    'import infinicore; from infinilm.lib import _infinilm; print("InfiniLM extension import: OK")'

GPU_COUNT=$(
    "$PYTHON" -c 'import torch; print(torch.cuda.device_count())'
)
if [ "$GPU_COUNT" -lt "$TP" ]; then
    echo "Expected at least $TP visible GPUs, found $GPU_COUNT" >&2
    exit 4
fi

if [ "$RUN_BENCH" = "1" ]; then
    "$BENCH_PYTHON" -c 'import vllm; print(f"vLLM client {vllm.__version__}")'
fi

if curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
    echo "A service is already listening at $HOST:$PORT; choose another PORT" >&2
    exit 5
fi

SERVER_PID=
cleanup() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
        echo "Stopping server process $SERVER_PID..."
        kill "$SERVER_PID" >/dev/null 2>&1 || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "Starting InfiniLM server..."
if [ "$SHOW_COMMANDS" = "1" ]; then
    print_command "${SERVER_COMMAND[@]}"
fi
"${SERVER_COMMAND[@]}" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

READY=0
ATTEMPT=0
START_SECONDS=$SECONDS
START_DEADLINE=$((START_SECONDS + SERVER_START_TIMEOUT))
while [ "$SECONDS" -lt "$START_DEADLINE" ]; do
    if curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
        READY=1
        break
    fi
    if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
        echo "Server exited before becoming ready" >&2
        tail -n 100 "$SERVER_LOG" >&2 || true
        exit 6
    fi
    ATTEMPT=$((ATTEMPT + 1))
    if [ $((ATTEMPT % 6)) -eq 0 ]; then
        echo "Waiting for model load and graph capture ($((SECONDS - START_SECONDS))s)..."
    fi
    sleep 5
done

if [ "$READY" != "1" ]; then
    echo "Server did not become ready within ${SERVER_START_TIMEOUT}s" >&2
    tail -n 100 "$SERVER_LOG" >&2 || true
    exit 7
fi

echo "Server is ready."
unset http_proxy https_proxy all_proxy ALL_PROXY || true

if [ "$RUN_BENCH" = "1" ]; then
    echo "Running the contest-style vLLM client..."
    if [ "$SHOW_COMMANDS" = "1" ]; then
        print_command "${BENCH_COMMAND[@]}"
    fi
    "${BENCH_COMMAND[@]}" 2>&1 | tee "$CLIENT_LOG"
else
    SMOKE_FILE="$RESULT_DIR/${RUN_NAME}-smoke.json"
    PAYLOAD=$(printf \
        '{"model":"%s","messages":[{"role":"user","content":"What is 15 + 27? Answer briefly."}],"max_tokens":%d,"temperature":0.0}' \
        "$MODEL_NAME" "$OUTPUT_LEN")
    curl -fsS "http://$HOST:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        --data-binary "$PAYLOAD" | tee "$SMOKE_FILE"
    printf '\n'
    "$PYTHON" - "$SMOKE_FILE" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as response_file:
    response = json.load(response_file)

try:
    choice = response["choices"][0]
    content = choice["message"]["content"]
    completion_tokens = response["usage"]["completion_tokens"]
except (KeyError, IndexError, TypeError) as error:
    raise SystemExit(f"Smoke validation failed: malformed response: {error}") from error

if not isinstance(content, str) or not content.strip():
    raise SystemExit("Smoke validation failed: empty completion")
if not isinstance(completion_tokens, int) or completion_tokens <= 0:
    raise SystemExit("Smoke validation failed: no completion tokens")
if "42" not in content:
    raise SystemExit("Smoke validation failed: arithmetic answer 42 not found")

print(
    "Smoke validation: PASS "
    f"(completion_tokens={completion_tokens}, answer_contains_42=yes)"
)
PY
fi

echo "Server log: $SERVER_LOG"
if [ "$RUN_BENCH" = "1" ]; then
    echo "Client log: $CLIENT_LOG"
    echo "Result JSON: $RESULT_DIR/$RESULT_FILE"
fi
