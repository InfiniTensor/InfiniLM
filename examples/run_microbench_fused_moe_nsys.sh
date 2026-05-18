#!/usr/bin/env bash
# Profile microbench_fused_moe_kernel.py with Nsight Systems (vendor vs upstream).
#
#   docker exec minicpm5-moe bash -lc 'export CUDA_VISIBLE_DEVICES=1; IMPL=vendor bash /home/zenghua/workspace/minicpm5-moe-support/InfiniLM/examples/run_microbench_fused_moe_nsys.sh'
#   docker exec minicpm5-moe bash -lc 'export CUDA_VISIBLE_DEVICES=1; IMPL=upstream bash /home/zenghua/workspace/minicpm5-moe-support/InfiniLM/examples/run_microbench_fused_moe_nsys.sh'
#
# Env:
#   IMPL=vendor|upstream   (default vendor)
#   OUT                    (default $REPO/InfiniLM/examples/bench_artifacts/nsys_microbench_moe_${IMPL})
#   ACTIVATION             (default silu)
#   NUM_TOKENS WARMUP ITERS
#
set -euo pipefail

REPO="${REPO:-/home/zenghua/workspace/minicpm5-moe-support}"
IMPL="${IMPL:-vendor}"
ACTIVATION="${ACTIVATION:-silu}"
NUM_TOKENS="${NUM_TOKENS:-128}"
WARMUP="${WARMUP:-2}"
ITERS="${ITERS:-12}"
ART="${REPO}/InfiniLM/examples/bench_artifacts"
OUT="${OUT:-${ART}/nsys_microbench_moe_${IMPL}}"
MB="${REPO}/InfiniLM/examples/microbench_fused_moe_kernel.py"
VPY="${VPY:-$REPO/.venv-vllm/bin/python}"

_torch_lib() {
  "$1" -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"
}

mkdir -p "$(dirname "$OUT")"
rm -f "${OUT}.nsys-rep" "${OUT}.sqlite"

SHAPE_ARGS=(
  --num-experts 32 --hidden 768 --intermediate 384 --top-k 4
  --num-tokens "$NUM_TOKENS"
  --seed 0
  --warmup "$WARMUP"
  --iters "$ITERS"
  --activation "$ACTIVATION"
  --dtype bfloat16
  --nvtx
)

if [[ "$IMPL" == "vendor" ]]; then
  export PYTHONPATH="$REPO/InfiniLM/python:$REPO/InfiniCore/python:${PYTHONPATH:-}"
  PY=(python3)
  unset LD_LIBRARY_PATH
  TL="$(_torch_lib python3)"
  export LD_LIBRARY_PATH="/root/.infini/lib:${TL}:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu"
  IMPL_ARG=(--impl infinilm)
elif [[ "$IMPL" == "upstream" ]]; then
  export PYTHONPATH="$REPO/InfiniLM/python:$REPO/InfiniCore/python:${PYTHONPATH:-}"
  PY=("$VPY")
  unset LD_LIBRARY_PATH
  TL="$(_torch_lib "$VPY")"
  export LD_LIBRARY_PATH="/root/.infini/lib:${TL}:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu"
  IMPL_ARG=(--impl vllm)
else
  echo "error: IMPL must be vendor or upstream, got $IMPL" >&2
  exit 1
fi

echo "[nsys-microbench] IMPL=$IMPL OUT=$OUT ACTIVATION=$ACTIVATION" >&2
if ! command -v nsys >/dev/null 2>&1; then
  echo "error: nsys not on PATH (install CUDA Nsight Systems)" >&2
  exit 1
fi

set -x
nsys profile -o "$OUT" --trace=cuda,nvtx,osrt --force-overwrite true \
  "${PY[@]}" "$MB" "${IMPL_ARG[@]}" --nvidia \
  "${SHAPE_ARGS[@]}"

set +x
echo "[nsys-microbench] wrote ${OUT}.nsys-rep" >&2
echo "[nsys-microbench] top kernels (cuda_gpu_kern_sum):" >&2
nsys stats --report cuda_gpu_kern_sum "${OUT}.nsys-rep" --timeunit ms 2>/dev/null | head -60 || true
