# Qwen3-235B true PPL CLI

This directory contains reproducible token-level perplexity tools for:

- Transformers BF16 on TP8
- InfiniLM BF16 on TP8
- InfiniLM W8A8 on TP8

The runners consume the same frozen token manifest and calculate causal,
shifted-token cross entropy:

```text
mean_nll = sum(-log p(x_t | x_<t)) / scored_token_count
ppl      = exp(mean_nll)
```

Long inputs use overlapping sliding windows, but every target token is scored
exactly once. Result JSON files include hashes of the manifest, token IDs and
scored indices. The comparison tools reject results whose workloads differ.

## Files

- `scripts/prepare_ppl_corpus_Qwen3_235B.py`: freeze UTF-8 text as a token manifest
- `scripts/infinilm/infinilm_ppl_Qwen3_235B.py`: InfiniLM BF16/W8A8 runner
- `scripts/transformers/pytorch_ppl_Qwen3_235B.py`: Transformers BF16 runner
- `scripts/calculate_true_ppl.py`: compare Transformers and InfiniLM results
- `scripts/calculate_infinilm_precision_ppl.py`: compare InfiniLM BF16 and W8A8
- `scripts/_ppl_common.py`: manifest, hashing and sliding-window logic
- `scripts/_gpu_guard.py`: refuse to start unless all eight GPUs are idle

## Prerequisites

The InfiniLM build must provide `InferEngine.score_nll`, and InfiniCore must
support FP16/BF16 logits with FP32 cross-entropy output. Run from Bash inside
the Hygon container. Set these paths for the current checkout and installation:

```bash
export INFINILM_ROOT=/path/to/InfiniLM
export INFINICORE_ROOT=/path/to/InfiniCore
export INFINICORE_LIB=/path/to/installed/infinicore/lib
export PPL_ROOT="$INFINILM_ROOT/test/ppl/qwen3_235b"
export MODEL_BF16=/data1/Qwen3_235B
export MODEL_W8A8=/data1/Qwen3_235B_quant
export TOKEN_MANIFEST=/path/to/wikitext2_raw_test_qwen3_235b.json
export LOG_DIR="$PWD/ppl_logs/$(date +%Y%m%d_%H%M%S)"

export PATH=/root/.local/bin:/opt/hyhal/bin:/opt/dtk/cuda/cuda/bin:/opt/dtk/bin:/opt/dtk/hip/bin:${PATH}
export PYTHONPATH="$PPL_ROOT/scripts:$INFINICORE_ROOT/python:$INFINILM_ROOT/python:/usr/local"
export LD_LIBRARY_PATH="$INFINICORE_LIB:/usr/local/lib/python3.10/dist-packages/torch/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/opt/ucx/lib:/opt/mpi/lib:/opt/hwloc/lib"
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

unset LIGHTOP_GPU_TARGET LIGHTOP_ASM_DIR
unset INFINILM_HYGON_LIGHTOP_DEVICE_NAME INFINILM_HYGON_LIGHTOP_NUM_CUS
unset INFINILM_HYGON_LIGHTOP_ALLOW_ASM INFINILM_LIGHTOP_CONFIG_DIR

mkdir -p "$LOG_DIR"
hy-smi --showpids
```

Do not start if another GPU workload is present. The runners repeat this check
and exit before loading the model when the GPUs are not idle.

## Prepare WikiText-2

Prepare the manifest once with the BF16 model tokenizer. All backends must use
this exact same manifest; do not tokenize the corpus separately per backend.

```bash
python -u "$PPL_ROOT/scripts/prepare_ppl_corpus_Qwen3_235B.py" \
  --input /path/to/wikitext-2-raw-v1-test.txt \
  --tokenizer "$MODEL_BF16" \
  --storage npy \
  --output "$TOKEN_MANIFEST"
```

Keep the generated JSON and adjacent `.tokens.npy` file together.

## Quick InfiniLM W8A8 check

This command scores only the first 127 target tokens. It validates model load,
`score_nll`, cross entropy and JSON output, but it is not a reportable full
WikiText-2 result.

```bash
set -o pipefail
timeout --signal=TERM --kill-after=60s 3600s \
  python -u "$PPL_ROOT/scripts/infinilm/infinilm_ppl_Qwen3_235B.py" \
    --model "$MODEL_W8A8" \
    --token-manifest "$TOKEN_MANIFEST" \
    --window 128 \
    --stride 64 \
    --max-scored-tokens 127 \
    --tp-size 8 \
    --attention flash-attn \
    --json-output "$LOG_DIR/infinilm_w8a8_smoke.json" \
  2>&1 | tee "$LOG_DIR/infinilm_w8a8_smoke.log"
rc=${PIPESTATUS[0]}
echo "INFINILM_W8A8_SMOKE_EXIT_CODE=$rc"
hy-smi --showpids
```

## Full WikiText-2 runs

`--max-scored-tokens 0` scores every target token in the manifest. Use the same
`window`, `stride` and `max-scored-tokens` values for every backend.

Transformers BF16:

```bash
set -o pipefail
timeout --signal=TERM --kill-after=60s 21600s \
  python -u "$PPL_ROOT/scripts/transformers/pytorch_ppl_Qwen3_235B.py" \
    --model "$MODEL_BF16" \
    --token-manifest "$TOKEN_MANIFEST" \
    --window 256 \
    --stride 128 \
    --max-scored-tokens 0 \
    --tp-size 8 \
    --attention eager \
    --json-output "$LOG_DIR/transformers_bf16_full.json" \
  2>&1 | tee "$LOG_DIR/transformers_bf16_full.log"
```

The Transformers entry point launches `torchrun` itself. Do not wrap it in a
second `torchrun` command. Eager attention is the validated Hygon path.

InfiniLM BF16:

```bash
set -o pipefail
timeout --signal=TERM --kill-after=60s 21600s \
  python -u "$PPL_ROOT/scripts/infinilm/infinilm_ppl_Qwen3_235B.py" \
    --model "$MODEL_BF16" \
    --token-manifest "$TOKEN_MANIFEST" \
    --window 256 \
    --stride 128 \
    --max-scored-tokens 0 \
    --tp-size 8 \
    --attention flash-attn \
    --json-output "$LOG_DIR/infinilm_bf16_full.json" \
  2>&1 | tee "$LOG_DIR/infinilm_bf16_full.log"
```

InfiniLM W8A8:

```bash
set -o pipefail
timeout --signal=TERM --kill-after=60s 21600s \
  python -u "$PPL_ROOT/scripts/infinilm/infinilm_ppl_Qwen3_235B.py" \
    --model "$MODEL_W8A8" \
    --token-manifest "$TOKEN_MANIFEST" \
    --window 256 \
    --stride 128 \
    --max-scored-tokens 0 \
    --tp-size 8 \
    --attention flash-attn \
    --json-output "$LOG_DIR/infinilm_w8a8_full.json" \
  2>&1 | tee "$LOG_DIR/infinilm_w8a8_full.log"
```

For a bounded formal run, replace `0` with the same positive token count in all
three commands, for example `10240`.

## Compare results

Transformers BF16 versus InfiniLM W8A8:

```bash
python -u "$PPL_ROOT/scripts/calculate_true_ppl.py" \
  --inputs \
    "$LOG_DIR/transformers_bf16_full.json" \
    "$LOG_DIR/infinilm_w8a8_full.json" \
  --max-ppl-increase-percent 20 \
  --json-out "$LOG_DIR/ppl_transformers_vs_w8a8.json"
```

InfiniLM BF16 versus InfiniLM W8A8:

```bash
python -u "$PPL_ROOT/scripts/calculate_infinilm_precision_ppl.py" \
  --inputs \
    "$LOG_DIR/infinilm_bf16_full.json" \
    "$LOG_DIR/infinilm_w8a8_full.json" \
  --max-ppl-increase-percent 20 \
  --json-out "$LOG_DIR/ppl_bf16_vs_w8a8.json"
```

Exit code `0` means the configured PPL increase threshold passed, `1` means it
failed, and `2` means the input files are invalid or describe different
workloads.

## Scope

PPL is a quality test. InfiniLM intentionally disables graph only for the
explicit `score_nll` path because it must retain full token logits/losses.
Normal generation and formal performance tests keep their existing graph path.
Do not report PPL scoring throughput as inference performance.
