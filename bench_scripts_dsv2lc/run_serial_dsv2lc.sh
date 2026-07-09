#!/usr/bin/env bash
set -u

MAX_WAIT_TIME="${MAX_WAIT_TIME:-1800s}"
SINGLE_GPU="${SINGLE_GPU:-0}"
TP2_GPUS="${TP2_GPUS:-0,1}"
MODEL="${MODEL:-/home_aclsylqidf/shared/DeepSeek-V2-Lite-Chat}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_TORCH="${RUN_TORCH:-1}"
RUN_INFINILM="${RUN_INFINILM:-1}"
PROGRESS_STEP="${PROGRESS_STEP:-128}"
REPEAT="${REPEAT:-3}"
export INFINILM_PRINT_ALL_BATCH_OUTPUTS="${INFINILM_PRINT_ALL_BATCH_OUTPUTS:-0}"

case "$REPEAT" in
  ''|*[!0-9]*|0)
    echo "REPEAT must be a positive integer, got: $REPEAT" >&2
    exit 2
    ;;
esac

OUT_DIR="${OUT_DIR:-serial_logs_$(date +%Y%m%d_%H%M%S)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO"
mkdir -p "$SCRIPT_DIR/$OUT_DIR"

CORE_LIB="${INFINICORE_LIB_DIR:-/home/libaoming/workplace/InfiniCore_latest/python/infinicore/lib}"
CORE_BUILD_LIB="${INFINICORE_BUILD_LIB_DIR:-/home/libaoming/workplace/InfiniCore_latest/build/linux/x86_64/release}"
export PYTHONPATH="/home/libaoming/workplace/InfiniCore_latest/python:$REPO:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$CORE_LIB:$CORE_BUILD_LIB:/opt/dtk/dcc/lib:/opt/dtk/aillvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/rccl/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/hyhal/lib:${LD_LIBRARY_PATH:-}"

DSV2_TORCH_DEVICE_MAP="${DSV2_TORCH_DEVICE_MAP:-dsv2_lmhead0}"
DSV2_TORCH_SPLIT_LAYER="${DSV2_TORCH_SPLIT_LAYER:-13}"
DSV2_TORCH_MAX_MEMORY="${DSV2_TORCH_MAX_MEMORY:-0:60GiB,1:60GiB}"
DSV2_TORCH_ATTN="${DSV2_TORCH_ATTN:-flash_attention_2}"
DSV2_INFINI_DEVICE="${DSV2_INFINI_DEVICE:-hygon}"
DSV2_INFINI_WEIGHT_LOAD="${DSV2_INFINI_WEIGHT_LOAD:-sync}"

log() {
  echo "[$(date '+%F %T')] $*"
}

run_cmd() {
  local log_file="$1"
  shift
  log "cmd: $*" | tee -a "$log_file"
  timeout --signal=TERM --kill-after=30s "$MAX_WAIT_TIME" "$@" >> "$log_file" 2>&1
}

run_torch_case() {
  local bs="$1" in_len="$2" out_len="$3" tp="$4"
  local case_name log_file devices status final_status run_idx
  printf -v case_name "dsv2lc_%s_%s_%s" "$bs" "$in_len" "$out_len"
  log_file="$SCRIPT_DIR/$OUT_DIR/torch_bench_$case_name.log"
  devices="$SINGLE_GPU"
  [ "$tp" = "2" ] && devices="$TP2_GPUS"

  log "[torch] start $case_name devices=$devices tp=$tp log=$log_file"
  {
    echo "[start] $(date '+%F %T') backend=torch case=$case_name devices=$devices tp=$tp"
    echo "[env] CUDA_VISIBLE_DEVICES=$devices HIP_VISIBLE_DEVICES=$devices"
  } > "$log_file"

  final_status=0
  for run_idx in $(seq 1 "$REPEAT"); do
    echo "[repeat_start] $(date '+%F %T') run=$run_idx/$REPEAT" >> "$log_file"
    if [ "$tp" = "2" ]; then
      CUDA_VISIBLE_DEVICES="$devices" HIP_VISIBLE_DEVICES="$devices" \
        run_cmd "$log_file" "$PYTHON_BIN" examples/pytorch_bench.py \
          --case-custom "$bs" "$in_len" "$out_len" \
          --model "$MODEL" \
          --devices "$devices" \
          --device-map "$DSV2_TORCH_DEVICE_MAP" \
          --split-layer "$DSV2_TORCH_SPLIT_LAYER" \
          --max-memory "$DSV2_TORCH_MAX_MEMORY" \
          --attn-implementation "$DSV2_TORCH_ATTN" \
          --last-token-logits \
          --progress-step "$PROGRESS_STEP" \
          --no-warmup
    else
      CUDA_VISIBLE_DEVICES="$devices" HIP_VISIBLE_DEVICES="$devices" \
        run_cmd "$log_file" "$PYTHON_BIN" examples/pytorch_bench.py \
          --case-custom "$bs" "$in_len" "$out_len" \
          --model "$MODEL" \
          --devices "$devices" \
          --device cuda \
          --attn-implementation "$DSV2_TORCH_ATTN" \
          --last-token-logits \
          --no-warmup
    fi
    status="$?"
    echo "[repeat_finish] $(date '+%F %T') run=$run_idx/$REPEAT status=$status" >> "$log_file"
    if [ "$status" != "0" ]; then
      final_status="$status"
      break
    fi
  done

  echo "[finish] $(date '+%F %T') status=$final_status" >> "$log_file"
  log "[torch] done $case_name status=$final_status"
  return "$final_status"
}

run_infinilm_case() {
  local bs="$1" in_len="$2" out_len="$3" tp="$4"
  local case_name log_file devices status final_status run_idx
  printf -v case_name "dsv2lc_%s_%s_%s" "$bs" "$in_len" "$out_len"
  log_file="$SCRIPT_DIR/$OUT_DIR/infinilm_bench_$case_name.log"
  devices="$SINGLE_GPU"
  [ "$tp" = "2" ] && devices="$TP2_GPUS"

  log "[infinilm] start $case_name devices=$devices tp=$tp log=$log_file"
  {
    echo "[start] $(date '+%F %T') backend=infinilm case=$case_name devices=$devices tp=$tp"
    echo "[env] CUDA_VISIBLE_DEVICES=$devices HIP_VISIBLE_DEVICES=$devices"
  } > "$log_file"

  final_status=0
  for run_idx in $(seq 1 "$REPEAT"); do
    echo "[repeat_start] $(date '+%F %T') run=$run_idx/$REPEAT" >> "$log_file"
    if [ "$tp" = "2" ]; then
      CUDA_VISIBLE_DEVICES="$devices" HIP_VISIBLE_DEVICES="$devices" \
        run_cmd "$log_file" "$PYTHON_BIN" examples/bench.py \
          --device "$DSV2_INFINI_DEVICE" \
          --use-mla \
          --enable-paged-attn \
          --weight-load "$DSV2_INFINI_WEIGHT_LOAD" \
          --model "$MODEL" \
          --batch-size "$bs" \
          --input-len "$in_len" \
          --output-len "$out_len" \
          --tp 2
    else
      CUDA_VISIBLE_DEVICES="$devices" HIP_VISIBLE_DEVICES="$devices" \
        run_cmd "$log_file" "$PYTHON_BIN" examples/bench.py \
          --device "$DSV2_INFINI_DEVICE" \
          --use-mla \
          --weight-load "$DSV2_INFINI_WEIGHT_LOAD" \
          --model "$MODEL" \
          --batch-size "$bs" \
          --input-len "$in_len" \
          --output-len "$out_len"
    fi
    status="$?"
    echo "[repeat_finish] $(date '+%F %T') run=$run_idx/$REPEAT status=$status" >> "$log_file"
    if [ "$status" != "0" ]; then
      final_status="$status"
      break
    fi
  done

  echo "[finish] $(date '+%F %T') status=$final_status" >> "$log_file"
  log "[infinilm] done $case_name status=$final_status"
  return "$final_status"
}

failures=0
log "output_dir=$SCRIPT_DIR/$OUT_DIR"
log "MAX_WAIT_TIME=$MAX_WAIT_TIME REPEAT=$REPEAT SINGLE_GPU=$SINGLE_GPU TP2_GPUS=$TP2_GPUS RUN_TORCH=$RUN_TORCH RUN_INFINILM=$RUN_INFINILM"

while read -r bs in_len out_len tp; do
  [ -z "$bs" ] && continue
  case "$bs" in \#*) continue ;; esac
  if [ "$RUN_TORCH" = "1" ]; then
    run_torch_case "$bs" "$in_len" "$out_len" "$tp" || failures=$((failures + 1))
  fi
  if [ "$RUN_INFINILM" = "1" ]; then
    run_infinilm_case "$bs" "$in_len" "$out_len" "$tp" || failures=$((failures + 1))
  fi
done <<'CASES'
4 1024 1024 1
4 4096 4096 1
16 128 128 1
16 1024 1024 1
16 4096 4096 2
CASES

log "summary failures=$failures output_dir=$SCRIPT_DIR/$OUT_DIR"
exit "$failures"
