### MiniCPM-SALA 16k long-prompt metrics (A/B cache modes)

**Setup**

- **Prompt construction**: `--target_input_tokens 16384` (actual synthesized **16386** chat-template tokens)
- **Workload**: `--max_new_tokens 1` (prefill-dominated)
- **Environment**: run via `scripts/run_compare_speed_in_container.sh` inside container `minicpm-sala`

| backend | cache_mode | attn_backend | enable_paged_attn | cache sizing | prefill_ttft_ms | prefill_throughput_tok_s | total_time_ms |
|---|---|---|---:|---|---:|---:|---:|
| hf | — | — | — | — | — | 9325.01 | 1757.21 |
| infinilm | static_fit | default | False | static_max_cache_len=16387 | 33632.05 | 487.21 | 33632.29 |
| infinilm | static_maxpos | default | False | static_max_cache_len=524288 | 34067.49 | 480.99 | 34067.75 |
| infinilm | paged | default | True | paged_block_size=256, paged_num_blocks=65 | 35626.25 | 459.94 | 35627.10 |

**Raw commands**

```bash
./scripts/run_compare_speed_in_container.sh --backends hf --target_input_tokens 16384 --max_new_tokens 1
./scripts/run_compare_speed_in_container.sh --backends infinilm --target_input_tokens 16384 --max_new_tokens 1 --infinilm_attn_backend default --infinilm_cache_mode static_fit
./scripts/run_compare_speed_in_container.sh --backends infinilm --target_input_tokens 16384 --max_new_tokens 1 --infinilm_attn_backend default --infinilm_cache_mode static_maxpos
./scripts/run_compare_speed_in_container.sh --backends infinilm --target_input_tokens 16384 --max_new_tokens 1 --infinilm_attn_backend default --infinilm_cache_mode paged --infinilm_paged_block_size 256
```

### Profiling methodology (nsys) for kernel attribution (HF vs InfiniLM prefill)

**Goal**: attribute the 16k prefill gap to kernel families (attention vs GEMMs vs layout/copies/sync), using the same prompt and a prefill-dominated workload.

**Environment**: all profiling commands in this section are run **inside the container `minicpm-sala`** (not on the host), so that PyTorch, InfiniCore, and the model path are available. Use `docker exec -it minicpm-sala bash` or the host script `./scripts/profile_prefill_torchprof_in_container.sh` to run in-container.

**Workload**

- HF: forward-only prefill (`--hf_mode forward_prefill`, `--max_new_tokens 1`)
- InfiniLM: prefill-dominated generation (`--target_input_tokens 16384 --max_new_tokens 1`)

**Key requirements**

- Use a free GPU to avoid allocator failures and noisy traces, e.g. `CUDA_VISIBLE_DEVICES=1`.
- Prefer `nsys stats` reports:
  - `cuda_gpu_kern_sum`
  - `cuda_gpu_mem_time_sum`
  - `cuda_api_sum`
  - `nvtx_sum`

**Example (inside container `minicpm-sala`)**

```bash
export CUDA_VISIBLE_DEVICES=1
REPO=/home/zenghua/workspace/minicpm-sala-support
MODEL=/data-aisoft/zenghua/models/OpenBMB/MiniCPM-SALA
OUT=${REPO}/profiles
mkdir -p ${OUT}

source /app/docker/nvidia/env-set.sh 2>/dev/null || true
export PYTHONPATH=${REPO}/InfiniLM/python:${REPO}/InfiniCore/python:${PYTHONPATH}

# HF forward-only prefill (single forward, best for kernel attribution)
nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt \
  -o ${OUT}/hf_forward_prefill_16k \
  python3 ${REPO}/InfiniLM/examples/compare_inference_speed.py \
    --model_path "${MODEL}" --prefill_16k --backends hf \
    --hf_mode forward_prefill --hf_forward_use_cache \
    --hf_forward_warmup 1 --hf_forward_iters 1 \
    --hf_attn_implementation flash_attention_2

# InfiniLM prefill-dominated (max_new_tokens=1)
nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt \
  -o ${OUT}/infinilm_prefill_16k \
  python3 ${REPO}/InfiniLM/examples/compare_inference_speed.py \
    --model_path "${MODEL}" --prefill_16k --backends infinilm \
    --infinilm_cache_mode static_fit --infinilm_attn_backend default

# Summaries
nsys stats --report cuda_gpu_kern_sum --format table ${OUT}/hf_forward_prefill_16k.nsys-rep > ${OUT}/hf_forward_prefill_16k_cuda_gpu_kern_sum.txt
nsys stats --report cuda_gpu_kern_sum --format table ${OUT}/infinilm_prefill_16k.nsys-rep   > ${OUT}/infinilm_prefill_16k_cuda_gpu_kern_sum.txt
nsys stats --report cuda_gpu_mem_time_sum --format table ${OUT}/hf_forward_prefill_16k.nsys-rep > ${OUT}/hf_forward_prefill_16k_cuda_gpu_mem_time_sum.txt
nsys stats --report cuda_gpu_mem_time_sum --format table ${OUT}/infinilm_prefill_16k.nsys-rep   > ${OUT}/infinilm_prefill_16k_cuda_gpu_mem_time_sum.txt
nsys stats --report cuda_api_sum --format table ${OUT}/hf_forward_prefill_16k.nsys-rep > ${OUT}/hf_forward_prefill_16k_cuda_api_sum.txt
nsys stats --report cuda_api_sum --format table ${OUT}/infinilm_prefill_16k.nsys-rep   > ${OUT}/infinilm_prefill_16k_cuda_api_sum.txt
nsys stats --report nvtx_sum --format table ${OUT}/hf_forward_prefill_16k.nsys-rep > ${OUT}/hf_forward_prefill_16k_nvtx_sum.txt
nsys stats --report nvtx_sum --format table ${OUT}/infinilm_prefill_16k.nsys-rep   > ${OUT}/infinilm_prefill_16k_nvtx_sum.txt
```

### Prefill kernel launch reduction: SiLU/SwiGLU evidence and change

**Evidence that SiLU/SwiGLU contributed to launch count**

- Prefill profiling (e.g. `profile_prefill_infinilm_torchprof.py` at seq_len=512) showed ~298k `cudaLaunchKernel` and many small **elementwise** kernels (~36k calls). The MLP path used two separate InfiniCore ops per layer for SwiGLU:
  - `infinicore::op::silu_(gate, gate)` — one kernel per layer
  - `infinicore::op::mul(gate, up)` — one kernel per layer
- With 32 layers that is **64 extra launches** from this pattern alone. InfiniCore provides a **fused** `swiglu(a, b)` (single kernel: `a * b * sigmoid(b)`), which matches SwiGLU as `silu(gate)*up` when called as `swiglu(up, gate)`.

**Change applied**

- **File**: `InfiniLM/csrc/models/minicpm_sala/minicpm_sala_mlp.cpp`
- **Before**: `silu_(gate, gate)` then `mul(gate, up)` (two kernel launches per layer).
- **After**: single `infinicore::op::swiglu(up, gate)` (one kernel per layer).
- **Effect**: 32 fewer kernel launches per prefill (one per layer). Re-run the same prefill profiler or nsys commands above and compare `cuda_api_sum` (e.g. `cudaLaunchKernel` count) and `cuda_gpu_kern_sum` to confirm.

### Environment fix: run InfiniLM/InfiniCore with InfLLM-v2 without LD_PRELOAD (nsys-safe)

When profiling with `nsys`, setting `LD_PRELOAD` to the `infllm_v2` extension can break `nsys` itself (loader errors from PyTorch's `libtorch_python.so`). To make `nsys profile ... python ...` work reliably, we preload the InfLLM-v2 `.so` **inside Python** (RTLD_GLOBAL) before importing `infinicore`, so that `libinfinicore_cpp_api.so` can resolve `mha_varlen_fwd` / `mha_fwd_kvcache` without using `LD_PRELOAD`.

- **Added helper**: `InfiniLM/python/infllmv2_loader.py`
- **Wired into scripts** (preload before `import infinicore`):
  - `InfiniLM/examples/compare_inference_speed.py`
  - `InfiniLM/examples/profile_prefill_infinilm_torchprof.py`
  - `InfiniLM/examples/minicpm_sala_logits_sanity.py`

This unblocks running both torchprof and `nsys profile` inside the `minicpm-sala` container with a consistent environment.

### 16k prefill nsys numbers (post env-fix)

**Workload:** `--prefill_16k` (prompt tokens 16386), `--max_new_tokens 1`, `--infinilm_cache_mode static_fit`, `--infinilm_attn_backend default`

- **HF forward-only prefill** (from `compare_inference_speed.py`): `total_time_ms ≈ 1782.58` for 16386 tokens.
- **HF forward-only prefill (rerun)** (from `compare_inference_speed.py`): `total_time_ms = 1757.21`, `prefill_throughput_tok_s = 9325.01` for 16386 tokens.
- **InfiniLM prefill-dominated** (from `compare_inference_speed.py`): `prefill_ttft_ms ≈ 55646.11` (baseline run) and `prefill_ttft_ms ≈ 57623.64` (rerun after minor code changes).

**InfiniLM 16k CUDA API summary** (nsys `cuda_api_sum`, baseline run `profiles/infinilm_prefill_16k_cuda_api_sum.txt`):

- `cudaLaunchKernel`: **3,147,266 calls**
- `cudaMemcpyAsync`: **394,155 calls**

Top GPU kernels by time (nsys `cuda_gpu_kern_sum`, baseline run `profiles/infinilm_prefill_16k_cuda_gpu_kern_sum.txt`) show very high call counts tied to the Lightning Simple GLA path:

- Several `at::native::*elementwise_kernel*` entries at **393,264 instances each** (exactly `16386 * 24`), indicating a large per-token kernel launch budget in the current GLA implementation.

**Prefill profiling: run inside container `minicpm-sala`**

All profiling commands below are intended to run **inside the container** (so PyTorch, InfiniCore, and the model are available). From the host you can either `docker exec -it minicpm-sala bash` and run the commands, or use the helper script that runs the torchprof prefill script in-container.

- **Launch-count confirmation (torchprof, in-container)**

  From repo root on host:

  ```bash
  ./scripts/profile_prefill_torchprof_in_container.sh
  ```

  Optional env: `SEQ_LEN=512` (default), `ACTIVE=1`, `MODEL_PATH`, `CUDA_VISIBLE_DEVICES`, `INFINILM_CUDA_INDEX`. The script prints `[launch_summary] cudaLaunchKernel_count=... cudaMemcpy_count=...` and the kernel table; compare after the SwiGLU fusion to confirm ~32 fewer launches per prefill.

  Or inside the container:

  ```bash
  source /app/docker/nvidia/env-set.sh 2>/dev/null || true
  export PYTHONPATH=${REPO}/InfiniLM/python:${REPO}/InfiniCore/python:${PYTHONPATH}
  cd ${REPO}/InfiniLM
  INFINILM_CUDA_INDEX=0 python3 examples/profile_prefill_infinilm_torchprof.py --model_path "${MODEL}" --seq_len 512 --active 1 --out /tmp/torchprof_prefill_512.txt
  ```

- **nsys prefill profiling** (see “Example (inside container minicpm-sala)” above) also runs in-container; use the same `REPO`, `MODEL`, `source env-set.sh`, and `PYTHONPATH` before `nsys profile` and `nsys stats`.
