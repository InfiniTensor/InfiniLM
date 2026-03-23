### MiniCPM-SALA long-context metrics + memory history

**Goal**: record reproducible long-context runs with:

- **time** (prefill TTFT / throughput)
- **peak GPU memory** (from 1s `nvidia-smi` polling)
- exact **command lines** and key env

**Notes**

- All commands are intended to run **inside** docker container `minicpm-sala`.
- Prefer an **idle** GPU (avoid indices that are already near full VRAM). Scan on the host (or `docker exec minicpm-sala nvidia-smi ...` if all GPUs are visible there):
  `nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits`
  Then set `export CUDA_VISIBLE_DEVICES=<N>` for the run and, for scripts that poll VRAM (e.g. `collect_metrics_longtext_decode.py`), set `NVML_GPU_INDEX=<N>` to the **same** index. Example when GPUs 2–4 are mostly free: `CUDA_VISIBLE_DEVICES=2` and `NVML_GPU_INDEX=2`.
- For InfiniLM + InfLLM-v2 builds, `libinfinicore_cpp_api.so` may require preloading `infllm_v2` with `RTLD_GLOBAL` before importing `infinicore`.

### OOM-safe sweep: one case per process

Running every long-context case in a **single** Python session can leave CUDA memory fragmented or peak across cases. Prefer `**run_longtext_metrics_cases.sh`**, which runs each `(backend × target × max_new)` as its **own** `python3 collect_metrics_longtext_decode.py --case ...` subprocess, appends one JSON line per row to `profiling_runs/longtext_decode_rows.jsonl`, then prints a markdown table via `--from-jsonl`.

```bash
REPO=/home/zenghua/workspace/minicpm-sala-support
export CUDA_VISIBLE_DEVICES=2
export NVML_GPU_INDEX=2
export PYTHONPATH=$REPO/InfiniLM/examples:$REPO/InfiniCore/python:$REPO/InfiniLM/python
export LD_LIBRARY_PATH=/root/.infini/lib:${LD_LIBRARY_PATH:-}
export METRICS_DATE=2026-03-23
cd $REPO/InfiniLM/examples
./run_longtext_metrics_cases.sh
```

Single case manually: `python3 collect_metrics_longtext_decode.py --case hf:16384` or `infinilm_rec:65536:1`. Monolithic (unsafe) full matrix: `python3 collect_metrics_longtext_decode.py --all-in-process`.

### Clean & Validate Status (post-cleanup: 2026-03-23)

- Clean: removed unused debug helper `log_tensor_stats_to_file_if_enabled` and deprecated metrics padded-decode cases; `collect_metrics_longtext_decode.py` + `run_longtext_metrics_cases.sh` no longer sweep `infinilm_pad:*`.
- Validate: rebuilt `_infinilm`, ran `InfiniCore/test/infinicore/ops/test_simple_gla_decode_recurrent.py --nvidia`, `test_simple_gla_prefill.py --nvidia`, and `InfiniLM/examples/minicpm_sala_logits_sanity.py` in `prefill` + `decode1`; confirmed `collect_metrics_longtext_decode.py --case infinilm_pad:*` is rejected with an "Unknown case kind" error.

---

## 2026-03-23 long-context + decode (`longtext_decode_rows.jsonl`)

Subprocess sweep via `./run_longtext_metrics_cases.sh`. **GPU:** `CUDA_VISIBLE_DEVICES=0`, `NVML_GPU_INDEX=0`. **Targets:** `METRICS_TARGETS=16384,32768`. **Decode steps:** `METRICS_DECODE_STEPS=32`. Recurrent InfiniLM uses `INFINI_LIGHTNING_GLA_RECURRENT_DECODE=1` with batched GLA state sync (GEMM). HF `total_ms` is end-to-end (prefill + decode), matching InfiniLM. `hf (decode_loop)` rows for `max_new=32` are appended via `hf:<target>:32`. Regenerate this table: `python3 collect_metrics_longtext_decode.py --from-jsonl profiling_runs/longtext_decode_rows.jsonl`


| date       | backend                                          | target_in | max_new | peak_mem_mib | total_ms | prefill_ttft_ms | prefill_tok_s | decode_itl_ms | decode_tok_s | gpu |
| ---------- | ------------------------------------------------ | --------- | ------- | ------------ | -------- | --------------- | ------------- | ------------- | ------------ | --- |
| 2026-03-23 | hf (decode_loop)                                 | 16384     | 1       | 38101        | 1821.53  | —              | —             | —             | —            | 0   |
| 2026-03-23 | hf (decode_loop)                                 | 32768     | 1       | 51545        | 3711.99  | —              | —             | —             | —            | 0   |
| 2026-03-23 | hf (decode_loop)                                 | 16384     | 32      | 38365        | 3435.09  | —              | —             | 52.24         | 19.14        | 0   |
| 2026-03-23 | hf (decode_loop)                                 | 32768     | 32      | 41717        | 5247.77  | —              | —             | 52.90         | 18.90        | 0   |
| 2026-03-23 | infinilm (static_fit, recurrent GLA decode)      | 16384     | 1       | 33525        | 3162.11  | 3161.5          | 5182.98       | —             | —            | 0   |
| 2026-03-23 | infinilm (static_fit, recurrent GLA decode)      | 32768     | 1       | 44897        | 7139.12  | 7138.74         | 4590.45       | —             | —            | 0   |
| 2026-03-23 | infinilm (static_fit, recurrent GLA, +32 decode) | 16384     | 32      | 33537        | 4111.32  | 3182.07         | 5149.48       | 29.94         | 33.4         | 0   |
| 2026-03-23 | infinilm (static_fit, recurrent GLA, +32 decode) | 32768     | 32      | 44911        | 8357.39  | 7146.78         | 4585.28       | 39            | 25.64        | 0   |


---

## History table


| date       | backend                                       | target_input_tokens | max_new_tokens | cache_mode | peak_mem_mib | total_time_ms | prefill_ttft_ms | prefill_throughput_tok_s | gpu |
| ---------- | --------------------------------------------- | ------------------- | -------------- | ---------- | ------------ | ------------- | --------------- | ------------------------ | --- |
| 2026-03-18 | hf                                            | 16384               | 1              | —          | 38091        | 1757.21       | —               | 9325.01                  | 2   |
| 2026-03-19 | hf                                            | 16384               | 1              | —          | 38091        | 1760.08       | —               | 9311.48                  | 2   |
| 2026-03-18 | hf                                            | 32768               | 1              | —          | 41173        | 3537.65       | —               | 9263.22                  | 2   |
| 2026-03-19 | hf                                            | 32768               | 1              | —          | 41151        | 3516.06       | —               | 9319.51                  | 2   |
| 2026-03-19 | infinilm(baseline)                            | 16384               | 1              | static_fit | 33570        | 2849.22       | 2849.03         | 5751.44                  | 0   |
| 2026-03-19 | infinilm(baseline)                            | 32768               | 1              | static_fit | 44174        | 5960.41       | 5960.14         | 5498.19                  | 0   |
| 2026-03-19 | infinilm(baseline)                            | 65536               | 1              | static_fit | 67195        | 13929.51      | 13929.12        | 4705.11                  | 4   |
| 2026-03-19 | hf (consistent-batch)                         | 16384               | 1              | —          | 38091        | 1782.63       | —               | 9192.04                  | 4   |
| 2026-03-19 | hf (consistent-batch)                         | 32768               | 1              | —          | 41173        | 3585.96       | —               | 9138.42                  | 4   |
| 2026-03-19 | hf (consistent-batch)                         | 65536               | 1              | —          | 47319        | 7426.98       | —               | 8824.32                  | 4   |
| 2026-03-19 | infinilm (consistent-batch)                   | 16384               | 1              | static_fit | 32605        | 2887.28       | 2887.06         | 5675.67                  | 4   |
| 2026-03-19 | infinilm (consistent-batch)                   | 32768               | 1              | static_fit | 43209        | 6005.78       | 6005.57         | 5456.60                  | 4   |
| 2026-03-19 | infinilm (consistent-batch)                   | 65536               | 1              | static_fit | 67195        | 13940.17      | 13939.90        | 4701.47                  | 4   |
| 2026-03-19 | infinilm (exp2/3 opt: strided KV + GLA views) | 32768               | 1              | static_fit | 38613        | 5993.70       | 5993.45         | 5467.64                  | 4   |
| 2026-03-19 | infinilm (exp2/3 opt: strided KV + GLA views) | 65536               | 1              | static_fit | 67195        | 13959.08      | 13958.78        | 4695.11                  | 4   |
| 2026-03-19 | infinilm(baseline)                            | 131072              | 1              | static_fit | 79883        | OOM           | —               | —                        | 6   |
| 2026-03-18 | hf                                            | 524288              | 1              | —          | 59591        | OOM           | —               | —                        | 3   |
| 2026-03-18 | hf                                            | 65536               | 1              | —          | 47319        | 7340.99       | —               | 8927.67                  | 1   |
| 2026-03-18 | hf                                            | 131072              | 1              | —          | 61641        | 15290.39      | —               | 8572.31                  | 1   |
| 2026-03-18 | hf                                            | 262144              | 1              | —          | 80059        | OOM           | —               | —                        | 1   |


---

## 2026-03-19 consistent batch summary (GPU 4, 1s polling)

Protocol used for both backends:

- same physical GPU (`CUDA_VISIBLE_DEVICES=4`), same model, `max_new_tokens=1`
- same target lengths: 16k / 32k / 64k
- memory measured from 1s `nvidia-smi -i 4 --query-gpu=memory.used` polling
- HF path: `--hf_mode forward_prefill --hf_forward_use_cache --hf_forward_warmup 1 --hf_forward_iters 1`
- InfiniLM path: `--infinilm_inprocess --infinilm_cache_mode static_fit`

### Growth deltas (16k->32k and 32k->64k)

TTFT note: HF forward-prefill does not emit TTFT; `total_time_ms` is used as prefill-time proxy for HF deltas.


| backend               | 16k->32k mem delta (MiB) | 32k->64k mem delta (MiB) | 16k->32k time delta (ms) | 32k->64k time delta (ms) |
| --------------------- | ------------------------ | ------------------------ | ------------------------ | ------------------------ |
| hf (forward-prefill)  | +3082                    | +6146                    | +1803.33                 | +3841.02                 |
| infinilm (static_fit) | +10604                   | +23986                   | +3118.51 (TTFT)          | +7934.33 (TTFT)          |


### Attribution profiling (InfiniLM 32k / 64k)

Artifacts are saved in `InfiniLM/examples/profiling_runs`:

- allocator logs: `alloc_infinilm_32768_gpu4.log`, `alloc_infinilm_65536_gpu4.log`
- nsys logs: `nsys_infinilm_32768_gpu4.log`, `nsys_infinilm_65536_gpu4.log`

Allocator observations (`INFINICORE_DEBUG_ALLOC=1`):

- both runs show identical small/medium allocation patterns (e.g., many `32 MiB` and `128 MiB` class allocations), suggesting these are mostly fixed/runtime-structural.
- 64k introduces substantially larger "large" allocations than 32k (examples in logs include `12.0 GiB`, `9.0 GiB`, and `2.0 GiB`-class requests), consistent with context-length-driven persistent KV slab growth.
- 32k large allocations are present but markedly smaller (e.g., `~6.0 GiB`, `~4.5 GiB`, `~1.0 GiB`), aligning with lower persistent cache footprint.

Nsight Systems observations (`nsys profile --trace=cuda,nvtx,osrt --stats=true`):

- NVTX `infinilm_generate` range scales from `~6.18s` (32k) to `~14.17s` (64k), matching TTFT growth.
- CUDA API summary becomes more memcpy-dominated at 64k:
  - 32k: `cudaMemcpy ~64.6%`, `cudaMemcpyAsync ~33.0%`
  - 64k: `cudaMemcpy ~83.0%`, `cudaMemcpyAsync ~15.7%`
- GPU kernel summary shows both attention and GLA prefill kernels scaling up:
  - `flash_fwd_kernel` total: `~1.03s` -> `~4.09s`
  - `simple_gla_prefill_chunked_kernel` total: `~1.24s` -> `~2.45s`

Attribution confidence:

- **High**: persistent KV/cache-related allocations are the primary memory-growth driver from 32k to 64k.
- **Medium**: transient prefill compute/workspace growth contributes, but is secondary vs persistent slabs for memory.
- **Medium**: synchronization/memcpy behavior is a major TTFT growth contributor at 64k.

### Short-context decode profiling (Nsight Systems, vs HF)

**Artifacts** (under `InfiniLM/examples/profiling_runs/`):

- HF manual decode: `nsys_decode_hf_tok256_gpu4.log` (`--hf_mode decode_loop`, short prompt, `max_new_tokens=256`).
- InfiniLM generate: `nsys_decode_infinilm_tok256_gpu4.log`, `nsys_decode_infinilm_nvtx_tok256_gpu4.log`, `nsys_decode_infinilm_nvtx_opt_tok256_gpu4.log` (same prompt / 256 new tokens; NVTX ranges from `infer_engine.generate`).
- Post–`write_i32`/`write_i64` rebuild (2026-03-20, GPU 4): `nsys_decode_infinilm_tok256_gpu4_pybind_run.log` (failed: stale `_infinicore` without `write_i32`), `nsys_decode_infinilm_tok256_gpu4_pybind_run2.log` + `decode_infinilm_tok256_gpu4_pybind_run2.nsys-rep` (**good** after `install.py` + `xmake build/install _infinicore` in container). Script `compare_inference_speed.py` preloads InfLLM-v2 (`RTLD_GLOBAL`) so `libinfinicore_cpp_api.so` resolves `mha_varlen_fwd`; bare `python -c import infinicore` without that preload can show an undefined-symbol error.

**NVTX (InfiniLM)** — use these ranges in the Nsight UI / `nsys stats` to isolate prefill vs steady decode:

- `infinilm_prefill_step` — first `generate` iteration.
- `infinilm_decode_total` — spans decode iterations 1..N-1 (opened on iter 1).
- `infinilm_decode_step` — one range per token step (high instance count).
- `infinilm_generate` — full `engine.generate()` call.

**HF**: `hf_decode_loop` wraps the timed decode loop (prefill is outside this range).

**Headline comparison** (same GPU, 256 decode steps, short prompt; numbers from the logs above):


| Metric (CUDA API sum)    | HF `decode_loop`    | InfiniLM `generate` |
| ------------------------ | ------------------- | ------------------- |
| `cudaLaunchKernel` calls | ~593k               | ~7.44M              |
| ~calls / decode step     | ~2.3k               | ~29k                |
| `cudaMemcpyAsync` calls  | lower than InfiniLM | ~988k               |


**Memcpy time** (`cuda_gpu_mem_time_sum`): InfiniLM decode shows large **H2D** wall share (~63% of memcpy time in one run) with **many** small transfers; HF decode shows **fewer** H2D operations but they can dominate memcpy time when they occur.

**Interpretation**: InfiniLM short decode is limited less by a single kernel and more by **per-step framework overhead** (launch count + small copies). Next wins are structural (fewer launches per token, true decode KV path, graph/capture where safe), not scalar metadata alone.

**Continuing profiling — repro commands** (inside `minicpm-sala`, pick idle `GPU`; outputs go to `profiling_runs/`):

```bash
REPO=/home/zenghua/workspace/minicpm-sala-support
MODEL=/data-aisoft/zenghua/models/OpenBMB/MiniCPM-SALA
GPU=4
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=$REPO/InfiniLM/python:$REPO/InfiniCore/python:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/root/.infini/lib:${LD_LIBRARY_PATH:-}
cd $REPO/InfiniLM/examples

TAG=decode_infinilm_tok256_gpu${GPU}
nsys profile --trace=cuda,nvtx,osrt --stats=true -o profiling_runs/${TAG} --force-overwrite true \
  python3 compare_inference_speed.py \
    --model_path "$MODEL" \
    --prompt "Write a short haiku about GPUs." \
    --max_new_tokens 256 \
    --backends infinilm \
    --no_hf \
    --infinilm_inprocess \
    --infinilm_cache_mode static_fit \
  2>&1 | tee profiling_runs/nsys_${TAG}.log

# Optional (InfiniLM decode): reduce D2H / Python overhead and A/B CPU metadata tensors
# export INFINI_PROFILE_KEEP_OUTPUT_IDS_ON_DEVICE=1
# export INFINI_PROFILE_COLLECT_OUTPUT_IDS=0
# export INFINI_PROFILE_DISABLE_FAST_DECODE_META=1   # force per-step from_list() metadata vs reusable CPU+write_i* fast path

TAG=decode_hf_tok256_gpu${GPU}
nsys profile --trace=cuda,nvtx,osrt --stats=true -o profiling_runs/${TAG} --force-overwrite true \
  python3 compare_inference_speed.py \
    --model_path "$MODEL" \
    --prompt "Write a short haiku about GPUs." \
    --max_new_tokens 256 \
    --backends hf \
    --no_infinilm \
    --hf_mode decode_loop \
    --hf_decode_warmup 8 \
    --hf_decode_iters 1 \
    --hf_attn_implementation flash_attention_2 \
  2>&1 | tee profiling_runs/nsys_${TAG}.log
```

**Long-context decode** (optional): add e.g. `--target_input_tokens 32768` to either command so NVTX still tags prefill vs decode; expect traces to be large.

**Prefill-only nsys** (matches earlier 32k/64k attribution):

```bash
TAG=infinilm_prefill_32768_gpu${GPU}
nsys profile --trace=cuda,nvtx,osrt --stats=true -o profiling_runs/${TAG} --force-overwrite true \
  python3 compare_inference_speed.py \
    --model_path "$MODEL" \
    --target_input_tokens 32768 \
    --max_new_tokens 1 \
    --backends infinilm \
    --no_hf \
    --infinilm_inprocess \
    --infinilm_cache_mode static_fit \
  2>&1 | tee profiling_runs/nsys_${TAG}.log
```

After code changes (e.g. pybind metadata path), re-run the **same** `TAG` with a suffix (`_run2`) and diff `cuda_api_sum` / `cuda_gpu_kern_sum` / NVTX tables.

### Ranked next optimization experiments (minimal changes)

1. **Constrain/reshape persistent KV growth first**
Expected impact: High memory reduction, likely best leverage on 32k->64k slope.
Minimal experiment: compare `static_fit` vs `paged` (small block sizes, e.g., 128/256) at 32k/64k and re-measure peaks + TTFT.
2. **Reduce transient prefill movement/workspace**
Expected impact: Medium TTFT gain, small-to-medium memory relief.
Minimal experiment: isolate `simple_gla_prefill` transform/workspace path and reduce extra copies/format conversions; confirm via reduced `cudaMemcpy` share in nsys.
3. **Trim synchronization/copy overhead around prefill**
Expected impact: Medium TTFT gain at long context.
Minimal experiment: profile before/after removing avoidable sync points or host-device transfers in attention/prefill orchestration; success criterion is lower `cudaMemcpy` wall share with unchanged logits.

Applied (2026-03-19): removed `permute(...)->contiguous()` materialization for KV cache update and GLA prefill inputs in `minicpm_sala_attention.cpp` (pass strided views).
Result: 32k peak memory improved on GPU 4 (**43209 MiB → 38613 MiB**) with similar TTFT; 64k peak unchanged (dominated by persistent KV slabs).

Validation gate for each experiment:

- **Operator unit tests (CUDA) first** — InfLLM-v2 + Simple GLA prefill (see below). Failing ops almost always mean wasted time on full-model logits debugging.
- run `minicpm_sala_logits_sanity.py` (prefill mode) and compare ratio/max_diff/mean_diff against current baseline.
- run one prompt generation sanity and verify no functional regression.

---

## Commands (repro)

### InfiniCore operator tests (run before logits sanity)

MiniCPM-SALA stack depends on `infllmv2_varlen` / `infllmv2_kvcache` and `simple_gla_prefill`. Run these inside `minicpm-sala` with `InfiniLM/python` on `PYTHONPATH` so InfLLM-v2 preloads before `import infinicore`:

```bash
REPO=/home/zenghua/workspace/minicpm-sala-support
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=$REPO/InfiniCore/test/infinicore:$REPO/InfiniCore/python:$REPO/InfiniLM/python:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/root/.infini/lib:${LD_LIBRARY_PATH:-}
cd $REPO/InfiniCore/test/infinicore/ops

python3 test_infllmv2_attention.py --nvidia
python3 test_simple_gla_prefill.py --nvidia
```

One-liner wrapper (same env assumptions as the repo):

```bash
bash $REPO/InfiniLM/examples/run_infinicore_ops_before_logits.sh
```

### Logits correctness gate (HF vs InfiniLM)

Run (inside `minicpm-sala`) to sanity-check HF vs InfiniLM prefill logits on a short prompt:

```bash
REPO=/home/zenghua/workspace/minicpm-sala-support
MODEL=/data-aisoft/zenghua/models/OpenBMB/MiniCPM-SALA
export CUDA_VISIBLE_DEVICES=1
export HF_CUDA_INDEX=0
export INFINILM_CUDA_INDEX=0
export PYTHONPATH=$REPO/InfiniLM/python:$REPO/InfiniCore/python:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/root/.infini/lib:${LD_LIBRARY_PATH:-}
cd $REPO/InfiniLM/examples

python3 minicpm_sala_logits_sanity.py \
  --model_path "$MODEL" \
  --mode prefill \
  --prompt "How are you? Tell me a short joke." \
  --k 10
```

Recorded output (2026-03-18, GPU=1):

```text
SANITY_ONELINE ratio=0.9889 max_diff=0.1875 mean_diff=0.0682
```

`--mode decode1` (prefill + one decode step): **prefill section** should match the prefill-only run. The **decode** section should now be finite (the previous `NaN` issue was traced to the CUDA embedding kernel leaving outputs uninitialized for out-of-range indices). Correctness can still diverge from HF for longer prompts due to decode/KV/attention parity work; treat **prefill** as the strongest HF parity gate for now.

### GPU scan

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits
```

### HF-only prefill (32k) with 1s memory polling

```bash
REPO=/home/zenghua/workspace/minicpm-sala-support
MODEL=/data-aisoft/zenghua/models/OpenBMB/MiniCPM-SALA
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=$REPO/InfiniLM/python:$REPO/InfiniCore/python:${PYTHONPATH:-}
cd $REPO/InfiniLM/examples

python3 compare_inference_speed.py \
  --model_path "$MODEL" \
  --target_input_tokens 32768 \
  --max_new_tokens 1 \
  --backends hf \
  --hf_mode forward_prefill \
  --hf_forward_use_cache \
  --hf_forward_warmup 1 \
  --hf_forward_iters 1 \
  --hf_attn_implementation flash_attention_2 \
  & pid=$!

echo "[mem] polling physical GPU 2 while pid=$pid"
while kill -0 $pid 2>/dev/null; do
  date +"%F %T"
  nvidia-smi -i 2 --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
  sleep 1
done
wait $pid
```

### InfiniLM-only (32k) with InfLLM-v2 preload + 1s memory polling

```bash
REPO=/home/zenghua/workspace/minicpm-sala-support
MODEL=/data-aisoft/zenghua/models/OpenBMB/MiniCPM-SALA
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=$REPO/InfiniLM/python:$REPO/InfiniCore/python:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/root/.infini/lib:$REPO/InfiniLM/build/linux/x86_64/release:${LD_LIBRARY_PATH:-}
cd $REPO/InfiniLM/examples

python3 - <<'PY' & pid=$!
import ctypes, os, runpy, sys
ctypes.CDLL("/usr/local/lib/python3.12/dist-packages/infllm_v2/C.cpython-312-x86_64-linux-gnu.so", mode=ctypes.RTLD_GLOBAL)
sys.argv = [
  "compare_inference_speed.py",
  "--model_path", os.environ["MODEL"],
  "--target_input_tokens", "32768",
  "--max_new_tokens", "1",
  "--backends", "infinilm",
  "--no_hf",
  "--infinilm_inprocess",
  "--infinilm_cache_mode", "static_fit",
]
runpy.run_path("compare_inference_speed.py", run_name="__main__")
PY

echo "[mem] polling physical GPU 2 while pid=$pid"
while kill -0 $pid 2>/dev/null; do
  date +"%F %T"
  nvidia-smi -i 2 --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
  sleep 1
done
wait $pid
```

