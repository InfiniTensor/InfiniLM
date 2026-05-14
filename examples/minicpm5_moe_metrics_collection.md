# MiniCPM5 MoE — metrics collection (single-prompt + e2e servers)

Living document: paste command output, JSON paths, or short tables after each run. Keep **workload knobs identical** when comparing backends unless the row explicitly notes a change.

**Builds and GPU issues:** run `xmake` / `install.py` for InfiniLM/InfiniCore **inside** the `minicpm5-moe` container so bind-mounted `.xmake` stays writable (see `.cursor/rules/minicpm5-docker-perf-workflow.mdc`). On **CUDA malloc / OOM**, switch `CUDA_VISIBLE_DEVICES` to another GPU after checking `nvidia-smi` memory use, then retry.

**Single-prompt smoke (required in this log):**

| Backend | Harness | Artifact (suggested) |
|---------|---------|----------------------|
| **InfiniLM** | `bench_balanced.py` | `bench_artifacts/single_prompt_infini_smoke.json` |
| **vLLM** | `vllm_bench_match_jiuge.py` | `bench_artifacts/single_prompt_vllm.json` |

**HF (`hf_bench_match_jiuge.py`)** is **out of scope here** (use `.venv-no-vllm` + `minicpm5_moe_inference_profiling.md` if you need Transformers 4.57.1 parity).

**Interpreter layout:**

| Role | Python | Notes |
|------|--------|--------|
| InfiniLM / `bench_balanced.py` | Image **`python3`** (or any env with built InfiniLM + matching CUDA torch) | `PYTHONPATH=$REPO/InfiniLM/python:$REPO/InfiniCore/python`; flash-attn / `LD_LIBRARY_PATH` per `.cursor/rules/minicpm5-docker-perf-workflow.mdc` |
| vLLM bench / vLLM serve | `$REPO/.venv-vllm` | `bash InfiniLM/examples/setup_vllm_venv.sh --moe` |
| vLLM MoE patch (workers) | `PYTHONPATH` | `$REPO/InfiniLM/examples/vllm_patches` + optional `MODEL`; see `vllm_patches/sitecustomize.py` |

Related: `minicpm5_moe_inference_profiling.md`, `bench_artifacts/` JSON snapshots.

---

## Session environment (fill each run)

| Field | Value |
|-------|-------|
| Date / UTC | 2026-05-13 (InfiniLM smoke re-run after `flash_attn` / `LD_PRELOAD` fixup) |
| Host / container | `minicpm5-moe` (Docker on host) |
| `CUDA_VISIBLE_DEVICES` | `5` (InfiniLM smoke row below) |
| `REPO` | `/home/zenghua/workspace/minicpm5-moe-support` |
| `MODEL` (checkpoint path) | `/data-aisoft/zenghua/models/minicpm5.16a3.v0314` |
| GPU name | `NVIDIA A100-SXM4-80GB` (`nvidia-smi -L`, index 5) |
| Driver / CUDA (if known) | Driver `580.105.08`; image CUDA matches NV PyTorch wheel (`cu128` in vLLM venv) |

**Interpreter versions**

| Track | `torch` | `transformers` / InfiniLM | `vllm` |
|-------|---------|---------------------------|--------|
| InfiniLM (`python3`) | `2.10.0a0+b4e4ee81d3.nv25.12` | N/A (InfiniLM in-tree) | N/A |
| `.venv-vllm` | `2.10.0+cu128` | `5.8.0` | `0.19.0` |

---

## 1) Single-prompt smoke (batch 1)

Align **prompt text** and **decode length** across InfiniLM and vLLM when comparing numbers. Default below: `Hi`, **7 prompt tokens** after chat template (match vLLM short smoke), **4** new tokens, greedy (`top_k=1`).

### 1.1 InfiniLM (`bench_balanced.py`) — **required**

Uses the same chat-template path as `jiuge.py` / `vllm_bench_match_jiuge.py` inside the harness. **Requires** InfiniCore + InfiniLM on `PYTHONPATH` and a working flash-attn stack if you pass `--attn flash-attn`.

```bash
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
REPO="${REPO:-/home/zenghua/workspace/minicpm5-moe-support}"
MODEL="${MODEL:-/data-aisoft/zenghua/models/minicpm5.16a3.v0314}"
export PYTHONPATH="$REPO/InfiniLM/python:$REPO/InfiniCore/python:${PYTHONPATH:-}"
TORCH_LIB="$(python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), \"lib\"))")"
FA="/usr/local/lib/python3.12/dist-packages/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so"
unset LD_LIBRARY_PATH
unset LD_PRELOAD
export LD_LIBRARY_PATH="/root/.infini/lib:$TORCH_LIB:/usr/local/lib/python3.12/dist-packages:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu"
cd "$REPO/InfiniLM/examples"
# Prefix-only LD_PRELOAD (do **not** `export LD_PRELOAD`): otherwise `tee`, `grep`, etc.
# inherit preload and hit `PyInstanceMethod_Type` / `libtorch_python.so` symbol errors.
LD_PRELOAD="$FA" python3 bench_balanced.py --nvidia \
  --model-path "$MODEL" \
  --prompt "Hi" \
  --prompt-tokens 7 \
  --max-new-tokens 4 \
  --top-k 1 \
  --attn flash-attn \
  --enable-paged-attn \
  --paged-kv-block-size 256 \
  --warmup-steps 3 \
  --runs 1 \
  --seed 0 \
  --json-out "$REPO/InfiniLM/examples/bench_artifacts/single_prompt_infini_smoke.json" \
  --print-json
```

| Field | Value | Notes |
|-------|-------|-------|
| `load_weights_s` | 212.88 | From JSON |
| `run_wall_s` | 0.75 | From JSON |
| `ttft_ms` | 633.48 | From JSON |
| `ttft_gpu_forward_ms` | 626.97 | From JSON |
| `avg_decode_itl_ms` | 38.71 | From JSON |
| `total_ms` | 749.60 | From JSON |
| `prompt_tokens_actual` | 7 | Matches vLLM tokenizer path |
| `max_new_tokens` | 4 | Greedy `top_k=1` |
| `generated_text` | (string) | New-token decode from the **last measured** run; written into `--json-out` for quick correctness checks |
| `generated_token_ids` | (list) | New-token ids for that same run |

**Artifact:** `InfiniLM/examples/bench_artifacts/single_prompt_infini_smoke.json`

**Docker (one copy-paste):** canonical smoke **plus** `INFINILM_FORCE_MOE_BACKEND=vllm_fused` and `baseline` reruns, with a safe summary (no nested heredocs). Set **`CUDA_VISIBLE_DEVICES` inside** `docker exec … bash -lc '…'` (a host-only `CUDA_VISIBLE_DEVICES=N docker exec …` prefix does **not** reliably select the GPU for processes inside the container):

```bash
docker exec minicpm5-moe bash -lc 'export CUDA_VISIBLE_DEVICES=1; bash /home/zenghua/workspace/minicpm5-moe-support/InfiniLM/examples/run_correctness_bench_smoke.sh'
```

Override paths inside the container if needed:

```bash
docker exec minicpm5-moe bash -lc 'export CUDA_VISIBLE_DEVICES=1 REPO=/home/zenghua/workspace/minicpm5-moe-support MODEL=/data-aisoft/zenghua/models/minicpm5.16a3.v0314; bash "$REPO/InfiniLM/examples/run_correctness_bench_smoke.sh"'
```

**Run notes / failures:**

- `flash_attn_preload.maybe_load_flash_attn_global()` strips `flash_attn_2_cuda*.so` from **process** `LD_PRELOAD` after `ctypes.CDLL(..., RTLD_GLOBAL)` so Triton/torch child shells do not preload the CUDA extension (avoids `undefined symbol: PyInstanceMethod_Type` on `/bin/sh`).
- Still use **prefix** `LD_PRELOAD="$FA" python3 ...` (not `export LD_PRELOAD`) so other commands in the same shell script (`tee`, `grep`, `wc`) never inherit that preload.
- **`LD_LIBRARY_PATH`** as in `.cursor/rules/minicpm5-docker-perf-workflow.mdc`.
- Harness defaults include **`--timing-discard-runs 1`** (drop one timed pass before the recorded run).
- **Cold load:** `load_weights_s` is dominated by first-time weight/materialization; not comparable to vLLM’s shorter second load in the same session without resetting caches.

---

### 1.2 vLLM (`vllm_bench_match_jiuge.py`)

**Interpreter:** `source "$REPO/.venv-vllm/bin/activate"`  
**Env:** `PYTHONPATH="$REPO/InfiniLM/examples/vllm_patches${PYTHONPATH:+:$PYTHONPATH}"`, `export MODEL=...`, `TORCHDYNAMO_DISABLE=1`.

```bash
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
REPO="${REPO:-/home/zenghua/workspace/minicpm5-moe-support}"
MODEL="${MODEL:-/data-aisoft/zenghua/models/minicpm5.16a3.v0314}"
export MODEL TORCHDYNAMO_DISABLE=1
export PYTHONPATH="$REPO/InfiniLM/examples/vllm_patches${PYTHONPATH:+:$PYTHONPATH}"
source "$REPO/.venv-vllm/bin/activate"
cd "$REPO/InfiniLM/examples"
python vllm_bench_match_jiuge.py \
  --model-path "$MODEL" \
  --prompt "Hi" \
  --max-new-tokens 4 \
  --json \
  --json-out "$REPO/InfiniLM/examples/bench_artifacts/single_prompt_vllm.json" \
  --gpu-memory-utilization 0.92 \
  --max-model-len 4096
```

For this checkpoint (~28 GiB weights on an 80 GiB A100), **low** `--gpu-memory-utilization` (e.g. `0.35`) leaves **no KV budget** after weight load (`ValueError: No available memory for the cache blocks`). Use a **high** fraction (e.g. `0.92`) and optionally shorten `--max-model-len` if you still hit limits. Always check `nvidia-smi` for a free GPU.

| Metric | Value | Notes |
|--------|-------|-------|
| `load_weights_s` | 40.69 | From JSON |
| `ttft_ms` | 145.37 | Wall TTFT |
| `prefill_engine_ms` | 112.49 | Engine monotonic interval |
| `decode_engine_ms` | 83.97 | 4 decode steps |
| `avg_decode_itl_ms` | 27.99 | Per-step average |
| `generate_wall_s` | 0.230 | End-to-end `generate()` |
| `prefill_tok_per_s` | 48.15 | |
| `decode_tok_per_s` | 35.73 | |
| `prompt_tokens` | 7 | |
| `n_generated` | 4 | |

**Artifact:** `InfiniLM/examples/bench_artifacts/single_prompt_vllm.json`

**Run notes / failures:**

- First attempt with `--gpu-memory-utilization 0.35` failed engine init (no KV blocks). **Successful** run: `--gpu-memory-utilization 0.92 --max-model-len 4096`.
- vLLM logs: `TransformersMoEForCausalLM` fallback path; HF cache `modeling_minicpm.py` emitted `cache_position` docstring warnings (non-fatal).

---

## 2) E2E server load (OpenAI-compatible `chat.completions`)

Use `InfiniLM/scripts/test_perf.py` (async client). It expects **`--base-url`** ending in `/v1` and a **`--model`** id that the server advertises.

**Useful flags (MiniCPM5 MoE):**

- *(Default)* omit `--prompt` — each request samples randomly from built-in Chinese **`PROMPTS`** in `test_perf.py`.
- `--prompt "…"` — override: use this user message for every request (including warmup).
- `--concurrency 1` — serial requests to isolate **server + streaming** path from queueing.
- `--warmup-requests N` — run `N` discarded streaming completions before the timed run (same prompt rule as the timed phase: fixed `--prompt` or random `PROMPTS` per completion); JSON includes `warmup_wall_s`. Optional `--warmup-max-tokens` overrides `max_tokens` for warmup only.
- JSON summary includes **`avg_decode_ms_per_chunk`** and **`avg_decode_wall_s`**: per-request `(elapsed − TTFT) / (# streamed text chunks)` averaged — closer to “decode ITL + HTTP/SSE” than **`avg_ms_per_stream_chunk`** (alias **`avg_ms_per_token`**): full stream wall divided by the same chunk count (mixes prefill, client `create` latency, and at **`--concurrency` > 1** other requests’ time between chunks on this stream). See **Glossary** below and `test_perf.py --help` epilog.

### Glossary: `test_perf.py` vs in-proc smoke

| JSON / print | What it measures | Compare to |
|--------------|-------------------|------------|
| `total_output_tokens` | Sum of **non-empty SSE `delta.content` events** (streaming text chunks), not HF tokenizer counts | — |
| `avg_ms_per_stream_chunk` (= legacy `avg_ms_per_token`) | Mean over requests of `(wall for full stream / #chunks)`; wall starts **before** `chat.completions.create` | Do **not** equate to `avg_decode_itl_ms` from `bench_balanced.py` under load |
| `avg_decode_ms_per_chunk` | Mean over requests of `(elapsed − TTFT) / #chunks` | **`avg_decode_itl_ms`** from [`bench_balanced.py`](InfiniLM/examples/bench_balanced.py) at **`--concurrency 1`** and short `--prompt` (e.g. §2.3) — typically same ballpark |
| `avg_ttft_s` | First SSE text delta minus wall clock at `create` start | Smoke `ttft_ms` (different boundaries) |

**Automated InfiniLM compare** (inside `minicpm5-moe`): run [`InfiniLM/examples/run_compare_metrics_and_concurrency.sh`](InfiniLM/examples/run_compare_metrics_and_concurrency.sh) — one server boot; `test_perf` c=1 `Hi`, then c=8 vs c=1 random `PROMPTS` load, then `bench_balanced` smoke; writes `bench_artifacts/compare_c1_decode_summary.json`, `compare_concurrency_summary.json`, and per-phase JSON. **Do not** `pkill -f infinilm...` from an inline `docker exec bash -lc '...'` that embeds that substring in the shell argv (it can kill the parent); use this script or `kill $server_pid`.

Install client deps in whichever venv you use to *drive* the benchmark (often `.venv-vllm` already has `openai` from vLLM):

```bash
python -m pip install openai
```

### 2.1 InfiniLM inference server (`python -m infinilm.server.inference_server`)

**Server start (paste your canonical command):**

```bash
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
REPO="${REPO:-/home/zenghua/workspace/minicpm5-moe-support}"
MODEL="${MODEL:-/data-aisoft/zenghua/models/minicpm5.16a3.v0314}"
export PYTHONPATH="$REPO/InfiniLM/python:$REPO/InfiniCore/python:${PYTHONPATH:-}"
TORCH_LIB="$(python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), \"lib\"))")"
unset LD_LIBRARY_PATH
unset LD_PRELOAD
export LD_LIBRARY_PATH="/root/.infini/lib:$TORCH_LIB:/usr/local/lib/python3.12/dist-packages:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu"
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
  --host 0.0.0.0
```

`inference_server` preloads FlashAttention via `InfiniLM/examples/flash_attn_preload.py` (no `LD_PRELOAD` of `flash_attn_2_cuda*.so` required).

**Served `base_url` / model id:**

| `base_url` (include `/v1`) | `model` id |
|----------------------------|------------|
| `http://127.0.0.1:8001/v1` | `minicpm5.16a3.v0314` (directory basename of `MODEL`) |

**Load probe:**

```bash
REPO="${REPO:-/home/zenghua/workspace/minicpm5-moe-support}"
python3 -m pip install -q "openai>=1.0"   # image python3 driver (once)
python3 "$REPO/InfiniLM/scripts/test_perf.py" \
  --base-url "http://127.0.0.1:8001/v1" \
  --model "minicpm5.16a3.v0314" \
  --num-requests 32 \
  --concurrency 8 \
  --max-tokens 128 \
  --warmup-requests 2 \
  --json-out "$REPO/InfiniLM/examples/bench_artifacts/e2e_infini_openai.json"
```

Without `--prompt`, each request picks a **random** line from `test_perf.py`’s `PROMPTS` pool. Re-run to refresh the snapshot below if the script or server changes (numbers vary run-to-run).

| Metric | Value |
|--------|-------|
| `requests_per_second` | 0.20 |
| `avg_latency_s` | 40.97 |
| `avg_ttft_s` | 0.64 |
| `avg_ms_per_token` | 324.97 |
| `avg_tokens_per_second` | 3.09 |
| `successful_requests` / `num_requests` | 32 / 32 |

**Artifact:** `InfiniLM/examples/bench_artifacts/e2e_infini_openai.json` (suggested path)

**Run notes:**

- Recorded in `minicpm5-moe` on **`CUDA_VISIBLE_DEVICES=7`** (2026-05-13): timed phase ~164 s wall for 32 requests + 2 warmups (~9 s). InfiniLM async server + streaming client is not yet tuned for high RPS vs vLLM; treat as a **baseline snapshot**, not a production SLA.
- Stop server when done: `pkill -f infinilm.server.inference_server` (or Ctrl+C in the foreground terminal).

### 2.1.1 MoE backend benefit (`baseline` vs `vllm_fused`, two server boots)

`INFINILM_FORCE_MOE_BACKEND` is read when the model is constructed, so compare backends by **restarting** the server with a different env (see [`run_e2e_server_moe_benefit.sh`](InfiniLM/examples/run_e2e_server_moe_benefit.sh)). Each JSON under `bench_artifacts/` includes `infinilm_moe_backend` after the run.

**One-shot:** set GPU **inside** `docker exec`:

```bash
docker exec minicpm5-moe bash -lc 'export CUDA_VISIBLE_DEVICES=1; bash /home/zenghua/workspace/minicpm5-moe-support/InfiniLM/examples/run_e2e_server_moe_benefit.sh'
```

**Session snapshot (2026-05-14, inner `export CUDA_VISIBLE_DEVICES=1`, A100, port `8016`, `num_requests=8`):** lighter load than the default §2.1 table (8 vs 32 requests); use for **relative** baseline vs fused only.

| Probe | `infinilm_moe_backend` | `requests_per_second` | `avg_ttft_s` | `avg_decode_ms_per_chunk` | `avg_latency_s` |
|--------|------------------------|------------------------:|-------------:|---------------------------:|----------------:|
| c=1, `--prompt "Hi"`, `max_tokens=4` | `baseline` | 1.77 | 0.41 | 51.36 | 0.56 |
| same | `vllm_fused` | 2.28 | 0.36 | 24.68 | 0.44 |
| c=8, random `PROMPTS`, `max_tokens=64` | `baseline` | 0.29 | 5.37 | 346.94 | 27.23 |
| same | `vllm_fused` | 0.35 | 0.58 | 352.60 | 22.80 |

**Concurrency:** `avg_decode_ms_per_chunk` stays large under **c=8** for both backends (streaming wall / chunk definition in `test_perf.py`); end-to-end **`avg_latency_s`** and **`avg_ttft_s`** still improved with `vllm_fused` in this snapshot. Remaining server-side batching / router CPU work is expected; see NVTX ranges in `minicpm5_moe_inference_profiling.md` and `MiniCPM5MoeVllmFusedSparseMoeBlock` for where time goes.

**Artifacts:** `e2e_infini_server_baseline_c1_hi.json`, `e2e_infini_server_baseline_c8.json`, `e2e_infini_server_vllm_fused_c1_hi.json`, `e2e_infini_server_vllm_fused_c8.json` in `InfiniLM/examples/bench_artifacts/`.

### 2.1.2 MoE **router** A/B (`vllm_fused`, GPU grouped topk vs `INFINILM_MOE_ROUTER=cpu`)

`INFINILM_MOE_ROUTER` is evaluated per forward (CPU path forces `run_router_topk_cpu` + pack); **`INFINILM_FORCE_MOE_BACKEND`** stays `vllm_fused`. Use **separate subprocesses** (smoke) or **server restarts** (e2e), same as backend switching.

**Smoke correctness** — [`run_correctness_bench_smoke.sh`](InfiniLM/examples/run_correctness_bench_smoke.sh): step 2 is **`vllm_fused`** with router default (GPU `minicpm5_grouped_sigmoid_topk` when registered); step 4 is **`vllm_fused` + `INFINILM_MOE_ROUTER=cpu`**. The script prints `generated_text_match` / `generated_token_ids_match` for the greedy `Hi` / 4 new tokens case.

**E2e benefit** — [`run_e2e_server_moe_benefit.sh`](InfiniLM/examples/run_e2e_server_moe_benefit.sh): after baseline and the first **`vllm_fused`** server (router default), a third server boot runs **`vllm_fused` + `INFINILM_MOE_ROUTER=cpu`** with the same `test_perf.py` probes. JSON files include `infinilm_moe_router` (`default` = unset, `cpu` = forced CPU router).

**One-shot (GPU inside `docker exec`):**

```bash
docker exec minicpm5-moe bash -lc 'export CUDA_VISIBLE_DEVICES=1; bash /home/zenghua/workspace/minicpm5-moe-support/InfiniLM/examples/run_correctness_bench_smoke.sh'
docker exec minicpm5-moe bash -lc 'export CUDA_VISIBLE_DEVICES=1; bash /home/zenghua/workspace/minicpm5-moe-support/InfiniLM/examples/run_e2e_server_moe_benefit.sh'
```

**Session snapshot (2026-05-14, inner `export CUDA_VISIBLE_DEVICES=1`, A100 80GB, port `8016`, `num_requests=8`):**

| Stage | Pass / note |
|-------|----------------|
| Smoke R-gpu vs R-cpu | **`generated_text_match=True`**, **`generated_token_ids_match=True`** (`single_prompt_infini_smoke_vllm_fused.json` vs `single_prompt_infini_smoke_vllm_fused_router_cpu.json`) |
| Regression guard | Same script still runs canonical + `vllm_fused` + `baseline` (steps 1–3) |

**E2E router-only (`vllm_fused`, same `test_perf` flags as §2.1.1):**

| Probe | `infinilm_moe_router` | `requests_per_second` | `avg_ttft_s` | `avg_decode_ms_per_chunk` | `avg_latency_s` |
|--------|------------------------|------------------------:|-------------:|---------------------------:|----------------:|
| c=1, `--prompt "Hi"`, `max_tokens=4` | `default` (GPU router) | 2.08 | 0.37 | 35.75 | 0.48 |
| same | `cpu` | 2.28 | 0.37 | 23.90 | 0.44 |
| c=8, random `PROMPTS`, `max_tokens=64` | `default` | 0.34 | 1.13 | 357.49 | 23.66 |
| same | `cpu` | 0.35 | 0.64 | 352.47 | 22.84 |

**Artifacts (router):** `single_prompt_infini_smoke_vllm_fused_router_cpu.json`; `e2e_infini_server_vllm_fused_router_cpu_c1_hi.json`, `e2e_infini_server_vllm_fused_router_cpu_c8.json` (R-gpu e2e reuses `e2e_infini_server_vllm_fused_c{1,8}.json` from §2.1.1).

### 2.1.3 E2E with **vLLM-imported** router (`INFINILM_MOE_ROUTER_ENGINE=vllm_poc`)

This path calls `vllm.model_executor.layers.fused_moe.router.grouped_topk_router.grouped_topk` from Python (see [`minicpm5_grouped_sigmoid_topk.py`](InfiniCore/python/infinicore/vendor/vllm_fused_moe/minicpm5_grouped_sigmoid_topk.py)). **Image `python3` has no `vllm`**, so the server must run under **`$REPO/.venv-vllm/bin/python`**. That venv’s Transformers stack failed **paged** KV startup (`invalid static kv cache config type`); the scripted run therefore uses **`--cache_type static`** and **`--attn default`** (venv usually has no `flash_attn`). **Do not compare these numbers directly to §2.1.1 / §2.1.2** (flash-attn + paged + system interpreter).

**Script:** [`run_e2e_server_moe_vllm_poc_router.sh`](InfiniLM/examples/run_e2e_server_moe_vllm_poc_router.sh) (default port **8017**). Ensure minimal server deps in the venv if imports fail, for example:

```bash
docker exec minicpm5-moe bash -lc 'REPO=/home/zenghua/workspace/minicpm5-moe-support; "$REPO/.venv-vllm/bin/pip" install -q janus xxhash fastapi uvicorn openai pydantic'
```

**One-shot:**

```bash
docker exec minicpm5-moe bash -lc 'export CUDA_VISIBLE_DEVICES=1; bash /home/zenghua/workspace/minicpm5-moe-support/InfiniLM/examples/run_e2e_server_moe_vllm_poc_router.sh'
```

**Session snapshot (2026-05-14, inner `CUDA_VISIBLE_DEVICES=1`, A100, port `8017`, `INFINILM_FORCE_MOE_BACKEND=vllm_fused`, `INFINILM_MOE_ROUTER_ENGINE=vllm_poc`, `attn=default`, `cache_type=static`, `num_requests=8`):**

| Probe | `requests_per_second` | `avg_ttft_s` | `avg_decode_ms_per_chunk` | `avg_latency_s` |
|--------|------------------------:|-------------:|---------------------------:|----------------:|
| c=1, `--prompt "Hi"`, `max_tokens=4` | 1.88 | 0.42 | 38.68 | 0.53 |
| c=8, random `PROMPTS`, `max_tokens=64` | 0.36 | 10.20 | 37.73 | 12.58 |

**Artifacts:** `e2e_infini_server_vllm_fused_router_engine_vllm_poc_c1_hi.json`, `e2e_infini_server_vllm_fused_router_engine_vllm_poc_c8.json` (JSON includes `infinilm_moe_router_engine`, `infinilm_e2e_python`, `infinilm_e2e_attn`, `infinilm_e2e_cache_type`).

**Nsight / NVTX (optional):** No capture this session. For kernel-level envelopes under R-gpu vs R-cpu, use ranges in [`minicpm5_moe_inference_profiling.md`](InfiniLM/examples/minicpm5_moe_inference_profiling.md) (`moe_vllm_fused::router_grouped_topk_gpu`, `moe_vllm_fused::router_d2h_cpu_topk_pack_h2d`, `moe_vllm_fused::fused_experts_dispatch`).

---

### 2.2 vLLM OpenAI server (`vllm serve`)

**Server start (paste your canonical command):**

```bash
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
REPO="${REPO:-/home/zenghua/workspace/minicpm5-moe-support}"
MODEL="${MODEL:-/data-aisoft/zenghua/models/minicpm5.16a3.v0314}"
export PYTHONPATH="$REPO/InfiniLM/examples/vllm_patches:$REPO/InfiniLM/python:$REPO/InfiniCore/python:${PYTHONPATH:-}"
export TORCHDYNAMO_DISABLE=1
source "$REPO/.venv-vllm/bin/activate"
vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port 8002 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.92 \
  --trust-remote-code \
  --enforce-eager \
  --served-model-name minicpm5.16a3.v0314
```

**Served `base_url` / model id:**

| `base_url` (include `/v1`) | `model` id |
|----------------------------|------------|
| `http://127.0.0.1:8002/v1` | `minicpm5.16a3.v0314` (`--served-model-name`) |

**Load probe:**

```bash
REPO="${REPO:-/home/zenghua/workspace/minicpm5-moe-support}"
source "$REPO/.venv-vllm/bin/activate"
python "$REPO/InfiniLM/scripts/test_perf.py" \
  --base-url "http://127.0.0.1:8002/v1" \
  --model "minicpm5.16a3.v0314" \
  --num-requests 32 \
  --concurrency 8 \
  --max-tokens 128 \
  --warmup-requests 2 \
  --json-out "$REPO/InfiniLM/examples/bench_artifacts/e2e_vllm_openai.json"
```

Same random-`PROMPTS` behavior as §2.1 when `--prompt` is omitted. Re-run to refresh the snapshot if needed.

| Metric | Value |
|--------|-------|
| `requests_per_second` | 4.20 |
| `avg_latency_s` | 1.41 |
| `avg_ttft_s` | 0.28 |
| `avg_ms_per_token` | 92.27 |
| `avg_tokens_per_second` | 5.12 |
| `successful_requests` / `num_requests` | 32 / 32 |

**Artifact:** `InfiniLM/examples/bench_artifacts/e2e_vllm_openai.json` (suggested path)

**Run notes:**

- Same GPU **`CUDA_VISIBLE_DEVICES=7`** as §2.1, **after** stopping InfiniLM (only one server at a time on this card).
- `test_perf.py` counts streamed **text** deltas as tokens; vLLM’s shorter `total_output_tokens` here vs InfiniLM reflects **early EOS / shorter completions** on the same random-`PROMPTS` mix, not a failed run (32/32 success).

---

### 2.3 Server overhead probe (`--concurrency 1`, `--prompt "Hi"`)

Isolates **HTTP + OpenAI streaming + server scheduler** from multi-client queueing. Same **`max_tokens`** as short smoke (4). Recorded in `minicpm5-moe`, **`CUDA_VISIBLE_DEVICES=7`**, with **`--warmup-requests 2`** before the timed phase (values in the table and JSON artifacts).

**InfiniLM** (server on `:8001`, then):

```bash
REPO="${REPO:-/home/zenghua/workspace/minicpm5-moe-support}"
python3 "$REPO/InfiniLM/scripts/test_perf.py" \
  --base-url "http://127.0.0.1:8001/v1" \
  --model "minicpm5.16a3.v0314" \
  --num-requests 16 \
  --concurrency 1 \
  --max-tokens 4 \
  --prompt "Hi" \
  --warmup-requests 2 \
  --json-out "$REPO/InfiniLM/examples/bench_artifacts/e2e_infini_openai_c1_short.json"
```

| Metric | Value | Notes |
|--------|-------|-------|
| `warmup_requests` | 2 | Discarded streaming completions |
| `warmup_wall_s` | 1.07 | Wall time for warmups only (not in timed RPS denominator) |
| `avg_ttft_s` | 0.35 | After warmup |
| `avg_decode_ms_per_chunk` | 23.37 | Compare to §1.1 `avg_decode_itl_ms` (~39 ms) |
| `avg_latency_s` | 0.42 | Full request wall (serial) |
| `avg_ms_per_token` | 140.05 | Still `elapsed/chunks`; lower after warmup but not decode ITL |
| `requests_per_second` | 2.38 | `16 / total_wall` for the timed phase only |

**Artifact:** `bench_artifacts/e2e_infini_openai_c1_short.json`

**vLLM** (server on `:8002`, `.venv-vllm` driver):

```bash
REPO="${REPO:-/home/zenghua/workspace/minicpm5-moe-support}"
source "$REPO/.venv-vllm/bin/activate"
python "$REPO/InfiniLM/scripts/test_perf.py" \
  --base-url "http://127.0.0.1:8002/v1" \
  --model "minicpm5.16a3.v0314" \
  --num-requests 16 \
  --concurrency 1 \
  --max-tokens 4 \
  --prompt "Hi" \
  --warmup-requests 2 \
  --json-out "$REPO/InfiniLM/examples/bench_artifacts/e2e_vllm_openai_c1_short.json"
```

| Metric | Value | Notes |
|--------|-------|-------|
| `warmup_requests` | 2 | |
| `warmup_wall_s` | 0.69 | |
| `avg_ttft_s` | 0.10 | |
| `avg_decode_ms_per_chunk` | 28.20 | Same order as §1.2 decode (~28 ms/step engine) |
| `avg_latency_s` | 0.19 | |
| `avg_ms_per_token` | 61.54 | |
| `requests_per_second` | 5.33 | |

**Artifact:** `bench_artifacts/e2e_vllm_openai_c1_short.json`

**Interpretation:** With **c=1**, **`Hi`**, and **two warmups**, InfiniLM **`avg_decode_ms_per_chunk`** (~23 ms) is close to in-proc smoke decode ITL; **`avg_ttft_s`** drops sharply vs no-warmup e2e because the first timed requests no longer pay one-off client/server cold path alone. **`avg_ms_per_token`** moves in the right direction after warmup but remains chunk-defined, not GPU-token ITL.

---

## 3) Cross-check summary (optional)

| Stage | InfiniLM in-proc | vLLM in-proc | InfiniLM server | vLLM server | Server c=1 / `Hi` |
|-------|-------------------|--------------|-------------------|-------------|-------------------|
| Single-prompt | §1.1 | §1.2 | — | — | — |
| E2E RPS / latency | — | — | §2.1 | §2.2 | §2.3 |

**Conclusions:**

- Single-prompt: `bench_artifacts/single_prompt_infini_smoke.json`, `bench_artifacts/single_prompt_vllm.json`.
- E2E OpenAI streaming (32× / c8 / Chinese): `bench_artifacts/e2e_infini_openai.json`, `bench_artifacts/e2e_vllm_openai.json`.
- E2E **overhead probe** (16× / c1 / `Hi` / `max_tokens` 4 / **`--warmup-requests 2`**): `bench_artifacts/e2e_infini_openai_c1_short.json`, `bench_artifacts/e2e_vllm_openai_c1_short.json` — use **`avg_decode_ms_per_chunk`** vs §1.1 decode ITL; **`avg_ms_per_token`** is chunk-based and still not 1:1 with GPU token ITL (§2.2).
