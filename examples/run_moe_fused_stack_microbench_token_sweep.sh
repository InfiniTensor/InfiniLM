#!/usr/bin/env bash
# Sweep --num-tokens for kernel-only vendor vs upstream fused_experts gap (synthetic MoE by default).
# Writes bench_artifacts/microbench_moe_fused_stack_gap_by_num_tokens.json (array of per-token gap objects).
#
#   docker exec minicpm5-moe bash -lc 'export CUDA_VISIBLE_DEVICES=1; bash /home/zenghua/workspace/minicpm5-moe-support/InfiniLM/examples/run_moe_fused_stack_microbench_token_sweep.sh'
#
# Env:
#   NUM_TOKENS_LIST="8 32 128 512"
#   MICROBENCH_GAP_USE_MODEL=1  — forwarded (real shapes from MODEL/config.json)
#   MICROBENCH_TUNED_CONFIG_DIR — forwarded to run_moe_fused_stack_microbench_gap.sh / microbench
#
set -euo pipefail

REPO="${REPO:-/home/zenghua/workspace/minicpm5-moe-support}"
export REPO
NUM_TOKENS_LIST="${NUM_TOKENS_LIST:-8 32 128 512}"
GAP_SH="$REPO/InfiniLM/examples/run_moe_fused_stack_microbench_gap.sh"
ART="$REPO/InfiniLM/examples/bench_artifacts"
OUT_JSON="$ART/microbench_moe_fused_stack_gap_by_num_tokens.json"
ROWS_TMP="$ART/.microbench_token_sweep_rows.jsonl"

mkdir -p "$ART"
rm -f "$ROWS_TMP"
touch "$ROWS_TMP"

for nt in $NUM_TOKENS_LIST; do
  echo "[token-sweep] num_tokens=$nt" >&2
  export NUM_TOKENS="$nt"
  bash "$GAP_SH" >&2 || { echo "[token-sweep] warn: gap script failed at num_tokens=$nt" >&2; continue; }
  single="$ART/microbench_moe_fused_stack_gap.json"
  if [[ ! -f "$single" ]]; then
    echo "[token-sweep] warn: missing $single" >&2
    continue
  fi
  python3 -c "
import json, os
d = json.load(open('$single', encoding='utf-8'))
d['num_tokens_sweep'] = int('$nt')
print(json.dumps(d, ensure_ascii=False))
" >>"$ROWS_TMP"
done

python3 - <<PY
import json, os

REPO = os.environ["REPO"]
rows_path = os.path.join(REPO, "InfiniLM", "examples", "bench_artifacts", ".microbench_token_sweep_rows.jsonl")
outp = os.path.join(REPO, "InfiniLM", "examples", "bench_artifacts", "microbench_moe_fused_stack_gap_by_num_tokens.json")
arr = []
if os.path.isfile(rows_path):
    with open(rows_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            arr.append(json.loads(line))
summary = {
    "note": "Each row: one run of run_moe_fused_stack_microbench_gap.sh at num_tokens_sweep. "
    "Isolates fused_experts vendor vs PyPI vLLM (not OpenAI server concurrency).",
    "microbench_gap_use_model": os.environ.get("MICROBENCH_GAP_USE_MODEL", "0"),
    "rows": arr,
}
with open(outp, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"[token-sweep] wrote {outp} ({len(arr)} rows)")
PY

echo "[token-sweep] DONE" >&2
