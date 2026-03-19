# MiniCPM-SALA on InfiniLM: Build Guide and Change Summary

This document describes the changes in **InfiniCore** and **InfiniLM** from their baseline commits to support MiniCPM-SALA with InfLLM-v2, the **prerequisites**, and a **step-by-step build and run guide**. With these changes, `InfiniLM/examples/jiuge.py` produces **reasonable MiniCPM-SALA generation output** when run with the correct environment.

**Baseline commits (for reference):**

- **InfiniLM:** `main`
- **InfiniCore:** `5fc85c8b1e6728839993f1b743a525a066da585f`

To see the exact diff from baseline:  
`git diff 5fc85c8b1e6728839993f1b743a525a066da585f -- InfiniCore` and  
`git diff main -- InfiniLM`.

---

## 1. Changes in InfiniCore (from `5fc85c8b1e6728839993f1b743a525a066da585f`)

InfiniCore was extended to **wire InfLLM-v2** (Stage-2 sparse attention) so that when built with `--infllmv2=<path>`, the C++ API calls `mha_varlen_fwd` and `mha_fwd_kvcache` from the infllmv2_cuda_impl .so.

### 1.1 New or modified files (summary)

| Area | Path | Purpose |
|------|------|--------|
| API (decl) | `include/infinicore/ops/infllmv2_api.hpp` | Declares `mha_varlen_fwd`, `mha_fwd_kvcache` (must be provided by infllmv2 .so at link/runtime). |
| API (decl) | `include/infinicore/ops/infllmv2_attention.hpp` | Public op header for infllmv2 attention. |
| Ops impl | `src/infinicore/ops/infllmv2_attention/infllmv2_attention.cc` | Implements `infllmv2_varlen` and `infllmv2_kvcache` by calling the above APIs when `ENABLE_INFLLMV2` and `ENABLE_ATEN` are set. |
| Pybind | `src/infinicore/pybind11/ops/infllmv2_attention.hpp` | Exposes infllmv2 ops to Python. |
| Pybind | `src/infinicore/pybind11/ops.hpp` | Includes infllmv2 op bindings. |
| Python | `python/infinicore/ops/infllmv2_attention.py` | Python wrapper for `infllmv2_varlen` / `infllmv2_kvcache`. |
| Python | `python/infinicore/__init__.py` | Exports `infllmv2_varlen`, `infllmv2_kvcache`. |
| Build | `xmake.lua` | New option `--infllmv2=<path>`; when set with `--aten=y`, defines `ENABLE_INFLLMV2` and links/rpath to the .so. |
| Test | `test/infinicore/ops/test_infllmv2_attention.py` | Unit tests for infllmv2 varlen/kvcache (skipped if not built or no CUDA). |
| Example | `examples/infllmv2_sanity.py` | Sanity script for InfLLM-v2 (skips if .so absent or no CUDA). |

### 1.2 Build option

- **Option:** `infllmv2` (path to directory containing a `.so` or to a single `.so` file).
- **Requires:** `aten=y` (InfiniCore must be built with PyTorch/ATen).
- **Effect:** Defines `ENABLE_INFLLMV2`, adds link and rpath to the infllmv2 .so. At runtime, `libinfinicore_cpp_api.so` resolves `mha_varlen_fwd` / `mha_fwd_kvcache` from that .so (via `LD_LIBRARY_PATH` or `LD_PRELOAD`).

---

## 2. Changes in InfiniLM (from `main`)

InfiniLM was extended to support the **MiniCPM-SALA** model (embedding, layers, attention, MLP, LM head) and to use InfiniCore (including InfLLM-v2 when available) for inference.

### 2.1 New or modified files (summary)

| Area | Path | Purpose |
|------|------|--------|
| C++ model | `csrc/models/minicpm_sala/*.cpp`, `*.hpp` | MiniCPM-SALA model: `minicpm_sala_attention`, `minicpm_sala_decoder_layer`, `minicpm_sala_model`, `minicpm_sala_for_causal_lm`, `minicpm_sala_mlp`. Per-layer dense KV cache; lightning (GLA) and optional InfLLM-v2 (minicpm4) attention paths. |
| C++ factory | `csrc/models/model_factory.cpp` | Registers MiniCPM-SALA model type. |
| Config | `python/infinilm/auto_config.py` | MiniCPM-SALA config handling. |
| Weights | `python/infinilm/modeling_utils.py` | MiniCPM-SALA weight loading (MuP scaling, etc.). |
| Examples | `examples/jiuge.py` | Generic InferEngine generation script; docstring updated with env (PYTHONPATH, LD_LIBRARY_PATH, LD_PRELOAD) for MiniCPM-SALA. |
| Examples | `examples/minicpm_sala_logits_sanity.py` | HF vs InfiniLM logits sanity (prefill/decode1/decodeN); single-token decode for correct KV cache; one-prompt output comparison. |
| Examples | `examples/modeling_minicpm_sala.py` | HF-side MiniCPM-SALA modeling (reference). |
| Docs | `MiniCPM_SALA_alignment_progress.md` | Alignment and debugging notes. |

### 2.2 Behaviour notes

- **Attention:** Layer 0 (minicpm4) can use compiled InfLLM-v2 when InfiniCore is built with `--infllmv2=` and the .so is preloaded; other layers use lightning (GLA) path.
- **Attention overhead optimizations:** In `minicpm_sala_attention.cpp`: (1) sequence lengths are read in one place when both `past_sequence_lengths` and `total_sequence_lengths` are present (`has_cache_meta`), avoiding duplicate logic; (2) Q/K/V use a single `contiguous()->view` chain after projections; (3) lightning path builds `q_bthd` via one `permute->contiguous` from `q_perm`; (4) sparse path uses `q_perm` directly (already contiguous) and only calls `contiguous()` on K/V when repeating heads. Semantics and logits are unchanged.
- **KV cache:** Decode must use **single-token input** per step; passing the full sequence each step would misalign the per-layer KV cache (see sanity script).
- **Engine / KV cache config:** MiniCPM-SALA uses per-layer dense KV cache in C++; the engine’s `cache_config` is used only for scheduling (e.g. `past_sequence_lengths` / `total_sequence_lengths`). **Static cache** is recommended (default in `jiuge.py` when not passing `--enable-paged-attn`). For static, `jiuge.py` sets `max_cache_len = max(initial_capacity, max_position_embeddings)` when `model_type == "minicpm_sala"` so long contexts are supported without re-alloc.

---

## 3. Prerequisites

### 3.1 System and toolchain

- **OS:** Linux.
- **Python:** 3.12 recommended (match the infllmv2 .so and InfiniCore pybind ABI).
- **CUDA:** 11.6+ (e.g. 12.x); `nvcc` in `PATH` (e.g. via `CUDA_HOME=/usr/local/cuda` and `PATH=$CUDA_HOME/bin:$PATH`).
- **C++:** GCC (e.g. `CC=gcc CXX=g++`) for infllmv2_cuda_impl and InfiniCore.
- **xmake:** For building InfiniCore (install from https://xmake.io or use a project-provided path).
- **PyTorch:** Installed in the same Python env used to build infllmv2 and to run InfiniLM (InfiniCore with `aten=y` links against this PyTorch’s libs).

### 3.2 Python environment

Use a **single venv** (or env) that has:

- `torch`
- `transformers`
- `triton` (e.g. 3.2.0; for MiniCPM-SALA HF path; if CUDA 12.8, a small patch may be needed for Triton’s `ptx_get_version` or use a Triton version that supports 12.8)
- `flash-linear-attention` (or HF deps for MiniCPM-SALA)
- Other InfiniLM/InfiniCore runtime deps

Build **infllmv2_cuda_impl** and **InfiniCore** with this same Python (and thus same PyTorch ABI).

### 3.3 Repo layout

- **minicpm-sala-support** (repo root) contains:
  - **InfiniCore/** — InfiniCore with InfLLM-v2 wiring.
  - **InfiniLM/** — InfiniLM with MiniCPM-SALA.
  - **sglang/3rdparty/infllmv2_cuda_impl/** — InfLLM-v2 CUDA kernel implementation (provides `mha_varlen_fwd`, `mha_fwd_kvcache`).

---

## 4. Build Guide

### 4.1 Build InfLLM-v2 (infllmv2_cuda_impl)

This produces the `.so` that provides `mha_varlen_fwd` and `mha_fwd_kvcache`. InfiniCore must be built with a PyTorch/ABI-compatible env (same Python/torch as here).

1. **From repo root:**
   ```bash
   cd sglang/3rdparty/infllmv2_cuda_impl
   ```
2. **Submodules:**
   ```bash
   git submodule update --init --recursive
   ```
3. **Env (recommended):**
   ```bash
   export CC=gcc CXX=g++
   export CUDA_HOME=/usr/local/cuda   # or your CUDA path
   export PATH=$CUDA_HOME/bin:$PATH
   ```
4. **Build/install** (use the Python that has torch and that you will use for InfiniLM):
   ```bash
   python setup.py install
   ```
   Or: `pip install -e .`
5. **Locate the .so:**  
   Typically under `build/lib.linux-x86_64-cpython-312/infllm_v2/` (name like `C.cpython-312-x86_64-linux-gnu.so`). Set:
   ```bash
   INFLLMV2_SO_DIR="<repo>/sglang/3rdparty/infllmv2_cuda_impl/build/lib.linux-x86_64-cpython-312/infllm_v2"
   ```

### 4.2 Build InfiniCore (with InfLLM-v2)

InfiniCore must be built with **aten** and, for MiniCPM-SALA with InfLLM-v2, **infllmv2** pointing at the .so (or directory containing it).

1. **Install Infini dependencies** (if not already):  
   Build and install Infini libs so they are under `$INFINI_ROOT` (default `~/.infini`). InfiniCore’s xmake expects `include/` and `lib/` there (e.g. `libinfinicore_cpp_api.so`, `libinfiniop.so`, etc.).

2. **From repo root:**
   ```bash
   cd InfiniCore
   ```
3. **Configure** (use the same Python/torch as infllmv2):
   ```bash
   xmake config -y --root --nv-gpu=y --aten=y --infllmv2="${INFLLMV2_SO_DIR}"
   ```
   Omit `--infllmv2=...` for a build without InfLLM-v2 (then no MiniCPM-SALA layer0 infllmv2 path).
4. **Build the Python extension:**
   ```bash
   xmake --root _infinicore
   ```
5. **Optional – install to ~/.infini:**
   ```bash
   xmake install
   ```
   The Python loadable is also copied under `InfiniCore/python/infinicore/lib/` by the build.

### 4.3 Run jiuge.py (MiniCPM-SALA)

Use the **same venv** that has `torch`, `transformers`, etc., and set env so InfiniCore and the infllmv2 .so are found and symbols resolve.

**Required:**

- `PYTHONPATH`: InfiniLM and InfiniCore Python packages.
- `LD_LIBRARY_PATH`: Torch lib, Infini lib (`/root/.infini/lib` or your `INFINI_ROOT/lib`), and optionally `INFLLMV2_SO_DIR` (if not using `LD_PRELOAD`).
- If InfiniCore was built with InfLLM-v2: **`LD_PRELOAD`** the infllmv2 .so so `libinfinicore_cpp_api.so` resolves `mha_varlen_fwd` (and `mha_fwd_kvcache`).

**Example (from repo root):**

```bash
INFLLMV2_SO_DIR="$(pwd)/sglang/3rdparty/infllmv2_cuda_impl/build/lib.linux-x86_64-cpython-312/infllm_v2"

PYTHONPATH="$(pwd)/InfiniLM/python:$(pwd)/InfiniCore/python:$PYTHONPATH" \
LD_LIBRARY_PATH="$(python -c 'import torch; print(torch.__path__[0])')/lib:/root/.infini/lib:${INFLLMV2_SO_DIR}:$LD_LIBRARY_PATH" \
LD_PRELOAD="${INFLLMV2_SO_DIR}/C.cpython-312-x86_64-linux-gnu.so" \
python InfiniLM/examples/jiuge.py --nvidia --model_path /root/.cache/modelscope/hub/models/OpenBMB/MiniCPM-SALA
```

Use the **venv** Python explicitly if needed, e.g.:

```bash
/path/to/venv/bin/python InfiniLM/examples/jiuge.py ...
```

For Triton (HF path) on CUDA 12.8 you may need:

```bash
TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
```

---

## 5. Verification

- **InfiniCore InfLLM-v2 ops:**  
  `PYTHONPATH=InfiniCore/python:InfiniCore/test/infinicore LD_LIBRARY_PATH=<torch_lib>:${INFLLMV2_SO_DIR}:/root/.infini/lib LD_PRELOAD=${INFLLMV2_SO_DIR}/C.cpython-312-x86_64-linux-gnu.so python InfiniCore/test/infinicore/ops/test_infllmv2_attention.py --nvidia`

- **HF vs InfiniLM logits (one-prompt decode):**  
  Same env + `LD_PRELOAD` and (if needed) `TRITON_PTXAS_PATH`:  
  `python InfiniLM/examples/minicpm_sala_logits_sanity.py --model_path <path> --mode decodeN --decode_steps 64`

- **Generation:**  
  `jiuge.py` with the same env should produce **reasonable MiniCPM-SALA output** (e.g. for prompt "How are you").

---

## 6. Related docs

- **CURRENT_PROGRESS.md** — Local progress, InfLLM-v2 plan, and run commands.
- **InfiniLM/MiniCPM_SALA_alignment_progress.md** — Alignment and debugging details.
- **sglang/3rdparty/infllmv2_cuda_impl/README.md** — InfLLM-v2 kernel design and install.
- **InfiniLM/examples/jiuge.py** — Docstring at top with env summary.

---

## 7. TODO

- **Remove temporal log and dump code** — Strip or gate debug logging, `INFINI_DEBUG_*`, and temporary dump paths (e.g. `/tmp/` tensor dumps, `dump_tensor_to_bin_if_enabled`, `log_tensor_stats_if_enabled`) from InfiniLM/InfiniCore once alignment and bring-up are stable.
- **Adapt inference_server.py** — Wire MiniCPM-SALA (and InfiniLM InferEngine) into the inference server (e.g. `inference_server.py` or equivalent in the workspace) so that the server can load and serve MiniCPM-SALA with the same env (PYTHONPATH, LD_LIBRARY_PATH, LD_PRELOAD) and run generation endpoints.

### 7.1 Debug and sanity env and code (for future erasing)

When removing temporal log and dump code, use this as the reference for **env parsing** and **locations to erase or gate**.

**Environment variables (debug / sanity):**

| Env var | Parsing / behavior | Purpose |
|---------|---------------------|--------|
| `INFINI_DEBUG_LOG` | Set to a file path (e.g. `/tmp/minicpm_sala_sanity_debug.log`). When set, C++ and Python append JSON/text lines to this file. | Text log for alignment debugging. |
| `INFINI_DEBUG_ATTN_DUMP` | Presence = enable (e.g. `"1"` or any). When set, tensors are written to fixed `/tmp/` paths below. | Enable binary tensor dumps and per-layer stats. |

**Where they are read:**

- **InfiniLM C++:** `std::getenv("INFINI_DEBUG_LOG")`, `std::getenv("INFINI_DEBUG_ATTN_DUMP")` in:
  - `InfiniLM/csrc/models/minicpm_sala/minicpm_sala_attention.cpp` (dump_tensor_f32, layer q/k/v/g_gamma and attn out dumps)
  - `InfiniLM/csrc/models/minicpm_sala/minicpm_sala_decoder_layer.cpp` (log_tensor_stats_if_enabled, tensor_to_f32_and_dump, layer input/out dumps)
  - `InfiniLM/csrc/models/minicpm_sala/minicpm_sala_model.cpp` (dump_tensor_to_bin_if_enabled, log_tensor_stats_if_enabled; embed and final hidden dumps)
- **InfiniLM Python (sanity script):** `os.environ["INFINI_DEBUG_LOG"]`, `os.environ["INFINI_DEBUG_ATTN_DUMP"]` set in `InfiniLM/examples/minicpm_sala_logits_sanity.py` before runs; `os.getenv("INFINI_DEBUG_*")` in `InfiniLM/examples/modeling_minicpm_sala.py` (HF-side hooks that write `/tmp/hf_*.pt` and log to `INFINI_DEBUG_LOG`).

**Temporary paths to remove or stop writing:**

- **C++ dumps (binary):** `/tmp/inf_embed_out.bin`, `/tmp/inf_final_hidden.bin`, `/tmp/inf_layer0_q.bin`, `/tmp/inf_layer0_k.bin`, `/tmp/inf_layer0_v.bin`, `/tmp/inf_layer0_g_gamma.bin`, `/tmp/inf_layer1_q.bin`, `/tmp/inf_layer1_k.bin`, `/tmp/inf_layer1_v.bin`, `/tmp/inf_layer1_g_gamma.bin`, `/tmp/inf_layer0_attn_input.bin`, `/tmp/inf_attn_out_layer0.bin`, `/tmp/inf_attn_out_layer1.bin`, `/tmp/inf_layer_out_<N>.bin`.
- **Python (sanity) writes:** `DEBUG_LOG_PATH` (e.g. `/tmp/minicpm_sala_sanity_debug.log`); `/tmp/hf_embed_out.pt`, `/tmp/hf_final_hidden.pt`, `/tmp/hf_layer0_attn_input.pt`, `/tmp/hf_layer_out_<idx>.pt`, `/tmp/hf_layer0_q.pt`, `/tmp/hf_layer0_k.pt`, `/tmp/hf_layer0_v.pt`, `/tmp/hf_attn_out_layer0.pt`, `/tmp/hf_layer1_q.pt`, `/tmp/hf_layer1_k.pt`, `/tmp/hf_layer1_v.pt`, `/tmp/hf_attn_out_layer1.pt`.
- **Helpers to remove or gate:** `dump_tensor_f32`, `dump_tensor_to_bin_if_enabled`, `log_tensor_stats_if_enabled`, `tensor_to_f32_and_dump`; sanity script’s `_append_debug_log`, and all `torch.save(..., "/tmp/...")` / `np.fromfile("/tmp/...")` / `os.path.isfile("/tmp/...")` blocks that exist only for alignment comparison.
