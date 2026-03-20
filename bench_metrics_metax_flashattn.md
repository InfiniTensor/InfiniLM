# MetaX flash-attn vs default benchmark (InfiniLM)

## Environment

- Device: MetaX / hpcc (inside `dev2`)
- Model: `/data-aisoft/zenghua/models/9g_8b_thinking_llama`
- Backend: `backend='cpp'`
- Batch size: `1`
- Tensor parallel: `1`
- Dtype: `bfloat16`
- KV cache: `enable_paged_attn=false` (static cache)
- Script: `InfiniLM/examples/bench.py`
- Measure: `total_time`, `Prefill TTFT`, `Decode Avg ITL`

## Recent summary (verified runs)

### Case 1: input_len=512, output_len=256


| Mode                    | total_time (ms) | Prefill TTFT (ms) / tok/s | Decode Avg ITL (ms) / tok/s |
| ----------------------- | --------------- | ------------------------- | --------------------------- |
| default                 | 10522.99        | 136.65 / 3746.92          | 40.72 / 24.56               |
| flash-attn              | 8864.50         | 96.79 / 5289.85           | 34.38 / 29.09               |
| delta (flash - default) | -15.73%         | -29.16% TTFT              | -15.55% ITL                 |


### Case 2: input_len=256, output_len=256


| Mode                    | total_time (ms) | Prefill TTFT (ms) / tok/s | Decode Avg ITL (ms) / tok/s |
| ----------------------- | --------------- | ------------------------- | --------------------------- |
| default                 | 8688.79         | 65.91 / 3884.18           | 33.81 / 29.58               |
| flash-attn              | 9697.21         | 74.78 / 3423.19           | 37.73 / 26.50               |
| delta (flash - default) | +11.59%         | +13.49% TTFT              | +11.57% ITL                 |


### Case 3: input_len=256, output_len=128


| Mode       | total_time (ms) | Prefill TTFT (ms) / tok/s | Decode Avg ITL (ms) / tok/s |
| ---------- | --------------- | ------------------------- | --------------------------- |
| default    | 4778.23         | 80.4 / 3184.09            | 36.99 / 27.04               |
| flash-attn | 4719.16         | 69.58 / 3679.48           | 36.61 / 27.32               |


delta (flash - default): **-1.22%** (slightly faster).ty

## Next benchmark sweep (longer sequences)

To compare performance for long contexts, we can sweep `input_len` and `output_len`.

Because generation time grows linearly with `output_len`, please confirm the sweep you want before we run up to very large values.

Suggested sweep options:

1. **Safe (recommended):** increase only `input_len` up to `65536`, keep `output_len` small (e.g. `256` or `1024`).
2. **Full (can be very slow):** increase both `input_len` and `output_len` up to `65536`.

Once you confirm, we’ll run `bench.py` for each (default, flash-attn) pair and append a new table section here.

## Longer input sweep (MetaX, output_len=1024)

Setting: `--metax --batch-size=1 --tp=1 --attn={default,flash-attn}`.

Measured metrics:

- `Prefill TTFT` (ms)
- `Decode Avg ITL` (ms)
- `total_time` (ms)


| input_len | default total_time (ms) | flash-attn total_time (ms) | delta   | default TTFT (ms) | flash TTFT (ms) | default ITL (ms) | flash ITL (ms) |
| --------- | ----------------------- | -------------------------- | ------- | ----------------- | --------------- | ---------------- | -------------- |
| 1024      | 36305.38                | 29936.58                   | -17.52% | 996.60            | 1037.57         | 34.51            | 28.25          |
| 2048      | 31283.06                | 33650.72                   | +7.55%  | 1057.20           | 1077.80         | 29.54            | 31.84          |
| 4096      | 38127.44                | 49012.66                   | +28.58% | 1505.09           | 1515.72         | 35.80            | 46.42          |
| 8192      | 40169.61                | 35903.26                   | -10.62% | 2634.85           | 2673.31         | 36.69            | 32.48          |
| 16384     | 45931.07                | 47083.18                   | +2.53%  | 6329.67           | 6238.41         | 38.71            | 39.92          |
| 32768     | OOM (hcMalloc)          | OOM (hcMalloc)             | -       | -                 | -               | -                | -              |
| 65536     | OOM (hcMalloc)          | OOM (hcMalloc)             | -       | -                 | -               | -                | -              |


Notes:

- For `input_len >= 32768` the run fails on MetaX with `hcMalloc(...) failed ... pinnable_block_allocator.cc`.
- So for `65536`, we can’t yet validate performance gain at `output_len=1024`; if you want, the next attempt should reduce `output_len` (e.g. 256/512/1024) to fit memory.

## Paged KV cache impact (`--enable-paged-attn`)

Even when you see “no impact” at some settings, `--enable-paged-attn` can significantly change the runtime behavior (cache manager + decode path).

Test setup (MetaX/dev2, bf16, batch=1,tp=1): `input_len=256`, `output_len=128`, `--warmup`.


| Attention backend | Paged KV cache | total_time (ms) | Prefill TTFT (ms) | Decode Avg ITL (ms) |
| ----------------- | -------------- | --------------- | ----------------- | ------------------- |
| `default`         | false          | 5178.45         | 68.19             | 40.23               |
| `default`         | true           | 3471.31         | 91.19             | 26.61               |
| `flash-attn`      | false          | 3834.37         | 66.06             | 29.67               |
| `flash-attn`      | true           | 6579.95         | 142.16            | 50.68               |


Takeaway from this run:

- `default` got faster with paged cache (~33% lower `total_time`)
- `flash-attn` got slower with paged cache (~72% higher `total_time`)

## Serving: InfiniLM `inference_server` + packaged `vllm bench serve` (dev2)

**Date:** 2026-03-20 (container `dev2`)

**Server env:**

```bash
REPO=/home/zenghua/workspace/fla-support
export PYTHONPATH=$REPO/InfiniLM/python:$REPO/InfiniCore/python:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/root/.infini/lib:/opt/conda/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
export HPCC_VISIBLE_DEVICES=0
```

**Server CLI (both runs; only `--attn` differs):**

```bash
python -m infinilm.server.inference_server \
  --metax \
  --model_path=/data-aisoft/zenghua/models/9g_8b_thinking_llama \
  --attn=default   # second run: --attn=flash-attn
  --host=0.0.0.0 --port=8000 \
  --max_tokens=2048 \
  --cache_type static \
  --max_cache_len=8192
```

**Client:** Conda `vllm` in dev2 (`vllm bench serve`, **not** the worktree `fla-support/vllm` — that checkout expects newer PyTorch than this image). Packaged benchmark uses `--endpoint-type openai-comp` and default `--endpoint /v1/completions`. InfiniLM maps `**/v1/completions`** to the same handler as chat and accepts `prompt` + streaming `max_tokens`.

**Bench CLI:**

```bash
mkdir -p /tmp/vllm_bench_infinilm
vllm bench serve \
  --endpoint-type openai-comp \
  --base-url http://127.0.0.1:8000 \
  --model 9g_8b_thinking_llama \
  --tokenizer /data-aisoft/zenghua/models/9g_8b_thinking_llama \
  --dataset-name random \
  --num-prompts 16 \
  --request-rate inf \
  --max-concurrency 4 \
  --random-input-len 512 \
  --random-output-len 128 \
  --trust-remote-code \
  --save-result \
  --result-dir /tmp/vllm_bench_infinilm \
  --result-filename bench_<attn>.json
```

**Workload:** random prompts, 512 input tokens, 128 output tokens, 16 requests, burst (`request-rate=inf`), max 4 concurrent.

### Results (vLLM bench summary)


| Server `--attn` | Duration (s) | Req/s | Output tok/s | Mean TTFT (ms) | Mean TPOT (ms) | Mean ITL (ms) |
| --------------- | ------------ | ----- | ------------ | -------------- | -------------- | ------------- |
| `default`       | 71.90        | 0.22  | 28.49        | 2473.05        | 121.49         | 121.49        |
| `flash-attn`    | 55.49        | 0.29  | 36.91        | 2151.93        | 91.71          | 91.71         |


JSON artifacts (inside `dev2`): `/tmp/vllm_bench_infinilm/bench_default_16.json`, `bench_flash_attn_16.json`.

**InfiniLM changes for this path:** `max_completion_tokens` alias in sampling params; `**/v1/completions`** route; streaming `**usage`-only** SSE chunk when `stream_options.include_usage` is set (needed for correct generated-token counts in packaged `openai-comp` client).

## Serving long I/O with vLLM bench serve (paged KV enabled) (dev2)

Test focus: longer input/output with paged KV cache enabled.

### Server

Only `--attn` differs between runs; cache config is the same:

```bash
python -m infinilm.server.inference_server \
  --metax \
  --model_path=/data-aisoft/zenghua/models/9g_8b_thinking_llama \
  --host=0.0.0.0 --port=8000 \
  --max_tokens=4096 \
  --cache_type paged \
  --max_batch_size=4 --num_blocks=512 --block_size=256 \
  --attn=default    # second run: --attn=flash-attn
```

### vLLM bench serve (fixed lengths)

```bash
mkdir -p /tmp/vllm_bench_long_paged
vllm bench serve \
  --endpoint-type openai-comp \
  --base-url http://127.0.0.1:8000 \
  --model 9g_8b_thinking_llama \
  --tokenizer /data-aisoft/zenghua/models/9g_8b_thinking_llama \
  --dataset-name random \
  --num-prompts 8 \
  --request-rate inf \
  --max-concurrency 2 \
  --random-input-len 8192 \
  --random-output-len 512 \
  --random-range-ratio 0 \
  --random-prefix-len 0 \
  --disable-tqdm \
  --save-result \
  --result-dir /tmp/vllm_bench_long_paged \
  --result-filename bench_paged_<attn>_8192_512.json
```

### Results (vLLM bench summary)

| `--attn`     | duration (s) | Req/s | Output tok/s | Mean TTFT (ms) | Mean TPOT (ms) | Mean ITL (ms) |
| ------------ | ------------- | ----- | ------------- | --------------: | --------------: | -------------: |
| `default`    | 89.49         | 0.089 | 23.77         | 774.91          | 92.42           | 81.42          |
| `flash-attn` | 88.79         | 0.090 | 23.96         | 1200.82         | 78.42           | 79.09          |

Artifacts (inside `dev2`):

- `/tmp/vllm_bench_long_paged/bench_paged_default_8192_512.json`
- `/tmp/vllm_bench_long_paged/bench_paged_flash_8192_512.json`

## Serving long I/O with worktree vLLM runner (paged KV enabled) (dev2)

This section re-runs the same long-I/O workload using the **worktree** vLLM code at `fla-support/vllm` by directly calling `vllm.benchmarks.serve.main_async` (no `vllm` CLI from the worktree). This avoids the container’s packaged vLLM client version skew.

### Workload (fixed lengths)

- `--dataset-name random`
- `--random-input-len 8192`
- `--random-output-len 512`
- `--num-prompts 4`
- `--max-concurrency 1`
- `--request-rate inf`

### Results (worktree runner JSON)

| `--attn` (server) | duration (s) | Req/s | Output tok/s | Mean TTFT (ms) | Mean TPOT (ms) | Mean ITL (ms) |
| ------------------ | ------------- | ----- | ------------- | --------------: | --------------: | --------------: |
| `default`          | 169.16        | 0.0236 | 12.11         | 11461.01        | 60.33           | 60.36           |
| `flash-attn`       | 108.61        | 0.0368 | 18.86         | 1510.73         | 50.18           | 50.28           |

Artifacts (inside `dev2`):

- `/tmp/vllm_bench_localvllm_runner_long/localvllm_paged_default_8192_512.json`
- `/tmp/vllm_bench_localvllm_runner_long_flash/localvllm_paged_flash_8192_512.json`

### Worktree vLLM runner (paged KV) with higher concurrency (conc=4)

Re-run with:

- `--dataset-name random`
- `--random-input-len 8192`
- `--random-output-len 512`
- `--num-prompts 8`
- `--max-concurrency 4`
- `--request-rate inf`
- `--num-warmups 2`

#### Results (worktree runner JSON, `max_concurrency=4`)

| `--attn` (server) | duration (s) | Req/s | Output tok/s | Mean TTFT (ms) | Mean TPOT (ms) | Mean ITL (ms) |
| ------------------ | ------------- | ----- | ------------- | --------------: | --------------: | --------------: |
| `default`          | 158.30        | 0.0505 | 25.87         | 45012.21         | 66.80           | 66.87           |
| `flash-attn`       | 67.29         | 0.1189 | 60.32         | 4220.47          | 58.09           | 58.13           |

Artifacts (inside `dev2`):

- `/tmp/vllm_bench_localvllm_runner_long_conc/localvllm_paged_default_conc4_8192_512.json`
- `/tmp/vllm_bench_localvllm_runner_long_conc/localvllm_paged_flash_conc4_8192_512.json`