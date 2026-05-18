#!/usr/bin/env bash
# Reproduce vendor vs upstream fused-experts overhead gap (same shapes, same seed) inside minicpm5-moe.
#
# Uses microbench_fused_moe_kernel.py:
#   --impl infinilm  → vendored Triton (same kernel family as INFINILM_MOE_FUSED_STACK=vendor experts)
#   --impl vllm      → PyPI vLLM fused_experts (same entrypoint as INFINILM_MOE_FUSED_STACK=upstream experts)
#
# Run from host (pick a **mostly free** GPU inside the container; index is local to the container):
#   docker exec minicpm5-moe bash -lc 'export CUDA_VISIBLE_DEVICES=1; bash /home/zenghua/workspace/minicpm5-moe-support/InfiniLM/examples/run_moe_fused_stack_microbench_gap.sh'
#
set -euo pipefail

REPO="${REPO:-/home/zenghua/workspace/minicpm5-moe-support}"
export REPO
MODEL="${MODEL:-/data-aisoft/zenghua/models/minicpm5.16a3.v0314}"
VPY="${VPY:-$REPO/.venv-vllm/bin/python}"
MB="$REPO/InfiniLM/examples/microbench_fused_moe_kernel.py"

_torch_lib() {
  "$1" -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"
}

# Avoid picking up a host Spack libstdc++ ahead of PyTorch (breaks ``_infinicore`` on vendor path).
_export_ld_for_torch() {
  local pyexe="$1"
  unset LD_LIBRARY_PATH
  local tl
  tl="$(_torch_lib "$pyexe")"
  export LD_LIBRARY_PATH="/root/.infini/lib:${tl}:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu"
}

# Full checkpoint MoE allocates large w1/w2; use synthetic dims by default so the gap repro
# fits a busy GPU. Set MICROBENCH_GAP_USE_MODEL=1 (and a free GPU) to bench real E/N/H from config.json.
# Optional: MICROBENCH_TUNED_CONFIG_DIR=/path/to/json_dir → --tuned-config-dir for both vendor and upstream legs.
MICROBENCH_GAP_USE_MODEL="${MICROBENCH_GAP_USE_MODEL:-0}"
# Optional: directory of vLLM-style ``E=...,N=...,device_name=....json`` (passed as ``--tuned-config-dir`` to both legs).
MICROBENCH_TUNED_CONFIG_DIR="${MICROBENCH_TUNED_CONFIG_DIR:-}"

WARMUP="${WARMUP:-10}"
ITERS="${ITERS:-40}"
NUM_TOKENS="${NUM_TOKENS:-128}"
SEED="${SEED:-0}"

export PYTHONPATH="$REPO/InfiniLM/python:$REPO/InfiniCore/python:${PYTHONPATH:-}"

echo "=== MoE fused stack microbench gap (vendor vs upstream experts) ===" >&2
echo "REPO=$REPO CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} MICROBENCH_GAP_USE_MODEL=$MICROBENCH_GAP_USE_MODEL MICROBENCH_TUNED_CONFIG_DIR=${MICROBENCH_TUNED_CONFIG_DIR:-}" >&2
if [[ "$MICROBENCH_GAP_USE_MODEL" == "1" ]]; then
  echo "MODEL=$MODEL (shapes from config.json)" >&2
else
  echo "MODEL=unused (synthetic MoE dims; export MICROBENCH_GAP_USE_MODEL=1 for full checkpoint)" >&2
fi
echo "num_tokens=$NUM_TOKENS warmup=$WARMUP iters=$ITERS seed=$SEED" >&2

if [[ ! -x "$VPY" ]]; then
  echo "error: venv python not found: $VPY" >&2
  exit 1
fi

if [[ "$MICROBENCH_GAP_USE_MODEL" == "1" ]]; then
  if [[ ! -f "$MODEL/config.json" ]]; then
    echo "error: missing config.json under MODEL=$MODEL" >&2
    exit 1
  fi
  shape_args=(--model-path "$MODEL")
else
  shape_args=(--num-experts 32 --hidden 768 --intermediate 384 --top-k 4)
fi

tuned_args=()
if [[ -n "${MICROBENCH_TUNED_CONFIG_DIR}" ]]; then
  tuned_args=(--tuned-config-dir "$MICROBENCH_TUNED_CONFIG_DIR")
fi

common_args=(
  --nvidia
  "${shape_args[@]}"
  --num-tokens "$NUM_TOKENS"
  --seed "$SEED"
  --warmup "$WARMUP"
  --iters "$ITERS"
  "${tuned_args[@]}"
  --print-json
)

echo "" >&2
echo "[1/2] vendor path: system python3 --impl infinilm …" >&2
_export_ld_for_torch python3
row_vendor="$(python3 "$MB" --impl infinilm "${common_args[@]}" 2>/dev/null | tail -1)"

echo "[2/2] upstream path: .venv-vllm --impl vllm …" >&2
_export_ld_for_torch "$VPY"
# vLLM logs tuned-config warnings on stderr; keep stderr out of command substitution.
row_upstream="$("$VPY" "$MB" --impl vllm "${common_args[@]}" 2>/dev/null | tail -1)"

echo "" >&2
echo "=== raw JSON rows ===" >&2
echo "$row_vendor" >&2
echo "$row_upstream" >&2

export _MBGAP_VENDOR_JSON="$row_vendor"
export _MBGAP_UPSTREAM_JSON="$row_upstream"
python3 - <<'PY'
import json, os

def _load(k: str) -> dict:
    raw = os.environ.get(k, "")
    if not raw.strip():
        raise SystemExit(f"missing or empty env {k!r}")
    return json.loads(raw)

v = _load("_MBGAP_VENDOR_JSON")
u = _load("_MBGAP_UPSTREAM_JSON")
for k in ("_MBGAP_VENDOR_JSON", "_MBGAP_UPSTREAM_JSON"):
    os.environ.pop(k, None)

mv, mu = v["cuda_ms_mean"], u["cuda_ms_mean"]
wv, wu = v["wall_ms_per_iter"], u["wall_ms_per_iter"]
gap_ms = mu - mv
gap_pct = 100.0 * (mu - mv) / mv if mv else float("nan")
gap_wall = wu - wv
gap_wall_pct = 100.0 * (wu - wv) / wv if wv else float("nan")
print("")
print("=== overhead gap (upstream − vendor) ===")
print(f"vendor_cuda_ms_mean:    {mv:.4f}")
print(f"upstream_cuda_ms_mean:  {mu:.4f}")
print(f"delta_cuda_ms:          {gap_ms:+.4f}  ({gap_pct:+.2f}% vs vendor)")
print(f"vendor_wall_ms_per_iter:   {wv:.4f}")
print(f"upstream_wall_ms_per_iter: {wu:.4f}")
print(f"delta_wall_ms:             {gap_wall:+.4f}  ({gap_wall_pct:+.2f}% vs vendor)")
print("")
print("Note: CUDA event timers can differ across torch builds; wall_ms_per_iter is host-side sanity.")
print("")
print("vendor_torch:   ", v.get("torch"))
print("upstream_torch: ", u.get("torch"))
print("cuda_device:    ", v.get("cuda_device"))

art = os.path.join(os.environ.get("REPO", "."), "InfiniLM", "examples", "bench_artifacts")
os.makedirs(art, exist_ok=True)
gap_path = os.path.join(art, "microbench_moe_fused_stack_gap.json")
out = {
    "vendor": v,
    "upstream": u,
    "delta_cuda_ms_mean": gap_ms,
    "delta_cuda_pct_vs_vendor": gap_pct,
    "delta_wall_ms_per_iter": gap_wall,
    "delta_wall_pct_vs_vendor": gap_wall_pct,
    "note": "Kernel-only fused_experts; vendor=infinilm impl, upstream=vllm impl (see microbench_fused_moe_kernel.py).",
}
with open(gap_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print("")
print(f"wrote {gap_path}")
PY
