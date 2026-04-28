# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

InfiniLM is an inference engine that bundles an inlined fork of [InfiniCore](https://github.com/InfiniTensor/InfiniCore). It provides LLM inference across multiple hardware backends (CPU, NVIDIA, Cambricon, Ascend, MetaX, Moore, Iluvatar, Kunlun, Hygon, Ali/QY). The repo composes four CMake subprojects — `runtime/` (`libinfinirt`), `ccl/` (`libinfiniccl`), `ops/` (`libinfinicore`), and a top-level pybind11 module `_infinilm` — plus the `infinilm` Python package.

## Prerequisites

- InfiniCore is **inlined** in this repo (subprojects under `runtime/`, `ccl/`, `ops/`); no separate InfiniCore install or `INFINI_ROOT` is needed for the new build path. `INFINI_ROOT` is only consulted by the legacy `scripts/libinfinicore_infer/` ctypes path.
- Submodules required: `git submodule update --init --recursive` (pulls `third_party/spdlog` and `third_party/json`).
- Python >= 3.10.

## Build Systems

**Two parallel build systems coexist** — pick the one that matches the target:

- **xmake (`xmake.lua`)** — builds the legacy `infinicore_infer` shared lib (from `src/`) for the `scripts/`-based ctypes path. Also has an `_infinilm` target, but the canonical way to build the pybind module is via CMake/`pip install -e .`.
- **CMake (`CMakeLists.txt`)** — drives `pip install -e .` via `setup.py`. Builds three vendored subprojects (`runtime/` → `libinfinirt.so`, `ccl/` → `libinfiniccl.so`, `ops/` → `libinfinicore.so` + `_infinicore` Python module) plus the `_infinilm` pybind11 module from `csrc/`. Co-locates all `.so`s under `python/infinilm/lib/` so RPATH=`$ORIGIN` works without `LD_LIBRARY_PATH`.

## Build Commands

The canonical build is CMake, driven from `setup.py`. `xmake.lua` is still in the tree but is legacy — prefer CMake.

```bash
# Legacy path: build/install infinicore_infer to $INFINI_ROOT
xmake && xmake install

# xmake build options
xmake f --use-kv-caching=true -cv      # KV-caching op (nvidia/ali/iluvatar/metax/hygon/qy)
xmake f --use-classic-llama=true -cv   # classic LlamaForCausalLM path

# Modern path: builds runtime/ccl/ops subprojects + _infinilm pybind module via CMake
pip install -e .

# Build C++ targets directly (without touching the Python package)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Build-time env knobs (consumed by setup.py)
INFINILM_BUILD_FLASH_ATTN=1 pip install -e .          # enable FlashAttention backend (auto-clones third_party/flash-attention)
INFINILM_FLASH_ATTN_ARCHS=80 pip install -e .         # restrict CUDA archs (default 80;86;89;90; single arch shrinks .so ~4×)
INFINILM_FLASH_ATTN_DIR=/abs/path pip install -e .    # use a pre-existing flash-attention checkout instead of cloning
INFINILM_BUILD_TYPE=Debug pip install -e .            # Debug | Release | RelWithDebInfo
INFINILM_BUILD_JOBS=8 pip install -e .                # parallel compile jobs (default nproc)
INFINILM_ENABLE_HYGON=1 pip install -e .              # build for Hygon DCU (DTK/HIP); see below
```

The CMake build also accepts `-DINFINILM_ENABLE_KV_CACHING=ON` and `-DINFINILM_USE_CLASSIC_LLAMA=ON` as the equivalents of the xmake flags above.

### Hygon DCU build

Hygon DCU uses the DTK toolchain (DTK 2604+, gfx906/gfx926/gfx928/gfx936/gfx938). DTK ships a complete CUDA-toolkit shim at `/opt/dtk/cuda/cuda-12/` (nvcc, cudart, cublas, cudnn, nccl-named `librccl`), so the build compiles the **same `.cu` sources** as NVIDIA via DTK's nvcc — only the `ENABLE_HYGON_API` define differs (which switches `infinirt::cuda` → `infinirt::hygon` and gates out FP8 / `cublasComputeType_t` paths the DCU can't handle).

```bash
INFINILM_ENABLE_HYGON=1 pip install -e . --no-build-isolation
```

`setup.py` auto-sources `/opt/dtk/env.sh` + `/opt/dtk/cuda/cuda-12/env.sh` when
`INFINILM_ENABLE_HYGON=1` is set (override the path via `INFINI_HYGON_DTK_ROOT`).
The runtime libtorch dependency is satisfied by `import torch` at the top of
`python/infinilm/__init__.py` — no manual `LD_LIBRARY_PATH` needed.

Implementation notes:

- The umbrella `INFINILM_ENABLE_HYGON=ON` flips `INFINIRT_ENABLE_NVIDIA` / `INFINICCL_ENABLE_NVIDIA` / `INFINIOPS_ENABLE_NVIDIA` off and the matching `_HYGON` options on (top-level `CMakeLists.txt`).
- `CMAKE_CUDA_ARCHITECTURES=75` is the right value: DTK's nvcc translates `sm_75` to the full DCU set internally.
- `CUDA_SEPARABLE_COMPILATION` is forced **OFF** under Hygon — DTK's device-link step drops gfx code from the final shared lib.
- RCCL is found via `find_library(rccl)` against `/opt/dtk/lib`.
- cuDNN works under DTK (libcudnn.so is shipped) so `INFINIOPS_ENABLE_CUDNN=ON` is fine.

Status: **all paths green on `/root/models/9g_8b_thinking_llama` tp=1, with `--block-size 64`**

| Configuration                    | Prefill    | Decode     |
|----------------------------------|------------|------------|
| Eager `--enable-paged-attn`      | 67 tok/s   | 55.6 tok/s |
| Eager `--attn=flash-attn`        | 63 tok/s   | 53.1 tok/s |
| Graph `--enable-graph` paged     | 233 tok/s  | 56.9 tok/s |
| Graph `--enable-graph` flash-attn| 236 tok/s  | 56.5 tok/s |

**`--block-size 64` is required for `--attn=flash-attn` correctness on gfx936.** The DTK
fork's `paged_attention` kernel silently half-writes output (every-other bf16 slot stays
at zero) for any other block size — no TORCH_CHECK, just garble. The companion symbol
`vllm_mha_fwd_kvcache` does enforce this via `TORCH_CHECK(block_size % 64 == 0, "Paged
KV cache block size must be divisible by 64 at gfx936 platform")`. Default block_size=256
(used by `--enable-paged-attn` alone, no FA) is fine for the handwritten `op::paged_attention_`
path but produces garble for the dlsym'd FA path. Setting `--block-size 64` also gives a
+35% decode boost for both paths (42 → 57 tok/s) since the kernel is tuned for it.

#### What it took beyond build wiring

Paged-attention required these source edits (all share the existing NVIDIA `.cu` kernels, gated for Hygon):
- Hygon entries in dispatch tables of `paged_attention/`, `paged_attention_prefill/`, `paged_caching/`, `embedding/` `operator.cc`.
- `ENABLE_HYGON_API` added to `paged_attention_prefill_nvidia.cu`'s top `#if`.
- `nvcuda::wmma` block in `paged_attention_prefill/cuda/kernel_v2.cuh` gated with `!defined(ENABLE_HYGON_API)`.
- `default_prefill_kernel()` returns `"warp"` under Hygon (DCU has no Tensor Cores → wmma kernel is unavailable).
- HYGON added to `embedding.cc` / `rmsnorm.cc` device whitelists.

Flash-attn (`ops/src/infinicore/adaptor/flash_attn_hygon_dlsym.cc`):
- Dlsym wrapper for `mha_fwd_kvcache` / `vllm_mha_varlen_fwd` / `paged_attention` from the system `flash_attn 2.8.3+das.opt1.dtk2604` wheel. Function-pointer types use `c10::optional` to match DTK ABI.
- `python/infinilm/__init__.py` `RTLD_GLOBAL`-preloads `flash_attn_2_cuda*.so` so symbols are visible to `dlsym(RTLD_DEFAULT, …)`.
- Decode path uses `paged_attention` (not `mha_fwd_kvcache`) — the latter calls `torch::empty` for `softmax_lse` per step which breaks HIP graph capture.
- K/V layout: K permuted to `[NB, nkv, BS, hd]`, V transposed to `[NB, nkv, hd, BS]`, both into pre-allocated scratches in `plan()`.
- `c10::hip::HIPStreamGuard` at the top of `run()` binds PyTorch's current stream to InfiniLM's stream so all ATen calls (including the dlsym'd FA kernels) participate in HIP graph capture.
- Compiled as a separate shared lib with plain g++ + `__HIP_PLATFORM_AMD__` so it can pull `<c10/hip/…>` headers without colliding with libinfinicore's DTK CUDA-compat include path.
- Lazy registration via explicit `infinicore_hygon_register_flash_attn()` called from `MhaKVCache` / `MultiheadAttentionVarlen` ctors — avoids static-init circular dep with libinfinicore.
- **Three non-obvious fixes** (only visible after diagnostic prints — neither matched the handoff's predicted root causes):
  1. `q.contiguous()` — q arrives as a non-contiguous slice of the fused QKV projection (`stride=[4608,128,1]` vs packed `[4096,128,1]`); DTK FA requires contiguous q.
  2. Compute `max_seqlen_q/k` from tensor shapes (`q.size(0)`, `block_table.size(1) * k.size(2)`) instead of using the caller-supplied `max_position_embeddings_` (65536 for this model). DTK FA uses these for internal workspace sizing — oversized values → VMFault.
  3. **Caller must use `--block-size 64`** — DTK fork's `paged_attention` kernel silently half-writes output for any other block size (default 256 produces every-other-zero bf16 pattern → garbled tokens from step 1). No runtime check fires. Sister symbol `vllm_mha_fwd_kvcache` enforces this via TORCH_CHECK; `paged_attention` does not. Affects all models (cross-model verified on Qwen2.5-7B and 9G-8B-thinking).

Graph mode "just worked" once flash-attn worked. The handoff's Bug 2a (ATen `index_select_out` for embedding) doesn't apply because our embedding op already uses a native CUDA kernel via `op::embedding` (with Hygon dispatch). Bug 2b (HIPStreamGuard) was already wired in the flash-attn TU.

## Running Inference

The new `examples/` + `python/infinilm/` path takes `--device <name>`; the legacy `scripts/` path takes `--<name>` flags. Devices: `cpu | nvidia | qy | metax | moore | iluvatar | ali | cambricon | hygon | ascend | kunlun`.

```bash
# Single inference (new path)
python examples/jiuge.py --device=nvidia --model=/path/to/model

# Distributed inference (tensor parallelism)
python examples/jiuge.py --device=nvidia --model=/path/to/model --backend=cpp --tp=4 --batch-size=16

# Experimental: warmup, paged attention, CUDA graph, attention backend
python examples/bench.py --device=nvidia --model=/path/to/model --warmup
python examples/bench.py --device=nvidia --model=/path/to/model --enable-paged-attn --enable-graph
python examples/bench.py --device=nvidia --model=/path/to/model --enable-paged-attn --attn=flash-attn

# Inference server (OpenAI-compatible API)
python python/infinilm/server/inference_server.py --device=nvidia --model=/path/to/model --tp=1

# Benchmark (C-Eval / MMLU). --cache-dir avoids HF network calls.
python test/bench/test_benchmark.py --device nvidia /path/to/model --bench ceval --subject all --backend cpp --tp 1 --output-csv results.csv

# Legacy path (scripts/, ctypes, --device-as-flag style)
python scripts/jiuge.py --nvidia /path/to/model 4
python scripts/launch_server.py --model /path/to/model --dev nvidia --ndev 4
python scripts/test_ppl.py --model /path/to/model
```

## Architecture

### C++ Core

Two C++ trees exist side-by-side, corresponding to the two build systems:

1. **Legacy `src/` → `infinicore_infer.so`** (xmake target). Public headers in `include/infinicore_infer/`. Contains model implementations (Jiuge/LLaMA variants, DeepSeek V3, Qwen3VL), tensor utilities, allocators, data loaders, cache manager. Consumed by `scripts/` via ctypes (`scripts/libinfinicore_infer/`).

2. **Modern `csrc/` → `_infinilm` pybind11 module** (CMake target, also has an xmake target). Layout:
   - `csrc/models/` — LLaMA-based model implementations using the `InfinilmModel` base class; new models register via `InfinilmModelFactory`.
   - `csrc/engine/` — `InferEngine` orchestrates distributed inference using `RankWorker` threads with barrier synchronization.
   - `csrc/cache/` — KV cache management, including paged-attention support.
   - `csrc/config/` — `ModelConfig` and `QuantConfig`.
   - `csrc/backends/` — Attention backend abstraction (default vs FlashAttention).
   - `csrc/layers/` — Shared layers (e.g., fused linear).
   - `csrc/global_state/`, `csrc/pybind11/` — global state and pybind glue.

3. **Vendored InfiniCore subprojects** (CMake-only; each has its own `CMakeLists.txt`):
   - `runtime/` → `libinfinirt.so` (runtime abstraction)
   - `ccl/` → `libinfiniccl.so` (collective communication)
   - `ops/` → `libinfinicore.so` + `_infinicore` Python module (operator library)
   These get co-located under `python/infinilm/lib/` so the wheel is self-contained.

**Deprecation note:** The legacy `InfinilmModel::Config`-based API is deprecated and scheduled for removal in v0.2.0 (Q2 2026). New code must use the `ModelConfig`-based polymorphic overloads.

### Python Package (`python/infinilm/`)

- `models/` — Python model wrappers
- `llm/` — LLM-specific logic
- `generation/` — Text generation pipeline
- `server/` — Inference server (OpenAI-compatible API)
- `distributed/` — Distributed/tensor-parallel support
- `cache/` and `cache_utils.py` — Cache management
- `infer_engine.py` — Python-side engine interface
- `lib/` — populated at build time with co-located `.so` files (do not commit)

`setup.py` also packages a sibling `infinicore` Python package (with `.nn`, `.ops` submodules) sourced from `python/`, exposed by the `_infinicore` module built from `ops/`.

### Tests (`test/`)

- `test/bench/test_benchmark.py` — C-Eval / MMLU accuracy benchmarks (see Running Inference above).
- `test/models/` — model-level test cases.

There is no top-level test runner wired up; invoke individual test scripts directly with `python`.

### Legacy Scripts (`scripts/`)

ctypes-based inference scripts that target the legacy `src/`-built `infinicore_infer.so` (loaded via `scripts/libinfinicore_infer/`):
- `jiuge.py`, `jiuge_awq.py`, `jiuge_gptq.py`, `jiuge_ppl.py` — Jiuge/LLaMA variants and quantized paths
- `deepseek.py`, `qwen3vl.py` — DeepSeek V3 and Qwen3-VL
- `launch_server.py`, `test_perf.py`, `test_ppl.py`, `test_ceval.py` — server launch and evaluation utilities
- `kvcache_pool.py`, `infer_task.py` — engine-side helpers

New work should target the `examples/` + `python/infinilm/` path; touch `scripts/` only when maintaining the legacy entry points.

## C++ Conventions

- C++17 standard, GCC toolchain
- Warnings treated as errors (`-Wall -Werror`)
- Namespace: `infinilm` (sub-namespaces: `engine`, `cache`, `config`, `backends`)
- Dependencies: inlined `infinicore` (from `ops/`), `infinirt` (from `runtime/`), `infiniccl` (from `ccl/`); spdlog and nlohmann/json under `third_party/`
