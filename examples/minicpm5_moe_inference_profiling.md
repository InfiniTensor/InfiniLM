# MiniCPM5 MoE — inference profiling (InfiniLM vs HF vs vLLM)

Use this document as a **continuous profiling** log: fill the environment block each session, then append or update the run table with fresh numbers. Workload settings should stay fixed when comparing engines.

**Structured metrics log:** for single-prompt smokes (**InfiniLM `bench_balanced.py` + vLLM**; HF optional elsewhere) and e2e OpenAI server runs, use **`minicpm5_moe_metrics_collection.md`** (tables + suggested JSON paths under `bench_artifacts/`).

---

## vLLM (profiling container)

**Use two dedicated Python environments:**

| Path | Role |
|------|------|
| **`$REPO/.venv-no-vllm`** | HF parity / `hf_bench_match_jiuge.py` — **`transformers==4.57.1`** (checkpoint default). Created with **`--system-site-packages`** so it reuses the container CUDA **torch** (no second torch install). Setup: `bash InfiniLM/examples/setup_hf_parity_venv.sh` |
| **`$REPO/.venv-vllm`** | vLLM + **`transformers>=5`** for `TransformersMoEForCausalLM` / `vllm_bench_match_jiuge.py`. **Standard** venv (no system-site-packages). Setup: `bash InfiniLM/examples/setup_vllm_venv.sh --moe` |

**Optional:** container **system `python3`** if the image already pins `transformers==4.57.1` — equivalent to the HF parity role, but a venv avoids accidental `pip install` upgrades on the image.

**One-command setup**

```bash
# HF parity (transformers 4.57.1, separate from vLLM):
bash InfiniLM/examples/setup_hf_parity_venv.sh
# or with Tsinghua mirror:
bash InfiniLM/examples/setup_hf_parity_venv.sh --tsinghua

# vLLM (isolated; MoE needs transformers>=5 in this venv only):
bash InfiniLM/examples/setup_vllm_venv.sh --moe
# or:
bash InfiniLM/examples/setup_vllm_venv.sh --moe --tsinghua
```

Or set a custom index for either script (examples):

```bash
export HF_PARITY_PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export HF_PARITY_PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
bash InfiniLM/examples/setup_hf_parity_venv.sh
```

**Do not** install vLLM into `.venv-no-vllm` or raise Transformers to 5.x there — that defeats HF parity.

---

### Manual vLLM venv (if not using ``setup_vllm_venv.sh``)

```bash
REPO=/home/zenghua/workspace/minicpm5-moe-support
python3 -m venv "$REPO/.venv-vllm"
source "$REPO/.venv-vllm/bin/activate"
python -m pip install -U pip
python -m pip install 'vllm==0.19.0'
```

**Pinned version:** `vllm==0.19.0` (session: 2026-04-14). Inside the venv this installs PyPI **`torch==2.10.0`** / **`torchvision==0.25.0`** / **`torchaudio==2.10.0`** (`+cu128` wheels) **only in `.venv-vllm`**, leaving **`.venv-no-vllm`** / system `python3` torch stacks separate.

**Runtime check (with venv active):**

```bash
source /home/zenghua/workspace/minicpm5-moe-support/.venv-vllm/bin/activate
python -c "import vllm, torch; print('vllm', vllm.__version__); print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

On success you should see `cuda True` and a device name from `torch.cuda.get_device_name(0)` when a GPU is visible.

**Every vLLM command** in this doc assumes `source "$REPO/.venv-vllm/bin/activate"` first (or an equivalent `python` from that venv).

### Isolated venv — avoid trashing the HF parity interpreter

Use a **standard** virtualenv for **`.venv-vllm`** (**do not** pass `--system-site-packages` there). Then:

- **`pip install` only mutates `$REPO/.venv-vllm`**.
- **`$REPO/.venv-no-vllm`** holds pinned **Transformers 4.57.1** for `hf_bench_match_jiuge.py` and HF parity; it uses `--system-site-packages` **only** to reuse CUDA **torch** from the image, not to mix in vLLM.

Never install vLLM into `.venv-no-vllm`.

### Trying MiniCPM5 MoE via **`TransformersMoEForCausalLM`** (Transformers fallback)

vLLM has **no native** MiniCPM5-MoE kernel path. For this checkpoint it **falls back** to the generic **`TransformersMoEForCausalLM`** backend (log line: *no vLLM implementation, falling back to Transformers implementation*). That path is what you are “trying” when you run vLLM with `trust_remote_code=True` on MiniCPM5.

**Inside `.venv-vllm` only**, after vLLM is installed (e.g. `setup_vllm_venv.sh` or manual `pip install 'vllm==0.19.0'`):

1. **Raise Transformers for the MoE backend** (still isolated; system Python unchanged). Skip if you already ran **`setup_vllm_venv.sh --moe`**.

   ```bash
   source "$REPO/.venv-vllm/bin/activate"
   python -m pip install 'transformers>=5.0.0,<6'
   ```

   Pip may warn that vLLM’s metadata prefers `transformers<5`; that is expected for this **experimental** fallback. All warnings apply **only** to the vLLM venv.

2. **Probe load** (real file, spawn-safe):

   ```bash
   python "$REPO/InfiniLM/examples/vllm_probe_load.py" \
     --model-path /path/to/minicpm5 \
     --max-model-len 8192 \
     --enforce-eager
   ```

   When the engine starts, logs should show **`TransformersMoEForCausalLM`** (Transformers MoE fallback).

3. **Benchmark** (single-prompt metrics):

   ```bash
   python -u "$REPO/InfiniLM/examples/vllm_bench_match_jiuge.py" \
     --model-path /path/to/minicpm5 \
     --prompt "Hi" --max-new-tokens 16 \
     --enforce-eager
   ```

   Add `--enforce-eager` first when remote code or MoE hits `torch.compile` issues.

4. **If it still fails** after load (e.g. `TransformersFusedMoE` / `len(self.experts)`), the checkpoint’s remote `modeling_*.py` is **incompatible** with vLLM’s fused expert wrapper; fixing that is a **model/vLLM integration** task, not an environment issue — your HF stack was never modified.

### MiniCPM5-MoE on vLLM (model gate)

| Item | Detail |
|------|--------|
| Checkpoint | `config.json` lists `architectures: ["MiniCPM5MoEForCausalLM"]`, `model_type: minicpm5_moe` |
| vLLM 0.19 behavior | Resolves to **`TransformersMoEForCausalLM`** (log: *no vLLM implementation, falling back to Transformers*). MoE path requires the Transformers modeling backend. |
| **Transformers pin tension** | vLLM **0.19.0** declares **`transformers>=4.56,<5`**, while **`TransformersMoEForCausalLM`** **requires `transformers>=5.0.0`**. **Preferred:** two venvs — **`.venv-no-vllm`** (HF / TF 4.57.1) vs **`.venv-vllm`** (vLLM MoE + TF≥5). |
| **Gate: Transformers 4.x** | With **4.57.1**, engine fails at model init: `ImportError: … requires transformers>=5.0.0 for MoE models support`. |
| **Gate: MiniCPM5 + Transformers 5 + vLLM 0.19** | Without patching, the checkpoint’s remote `modeling_minicpm.py` hits **`TypeError: object of type 'TransformersFusedMoE' has no len()`** inside MoE (`one_hot(..., num_classes=len(self.experts))`). |
| **Workaround (A1 patch, 2026-04-14)** | With the `sitecustomize` patch enabled (see `vllm_minicpm5_moe_patch.md`) and **`--enforce-eager`**, MiniCPM5 MoE **loads** and `vllm_bench_match_jiuge.py` can report **single-prompt TTFT / decode ITL**. |
| HF baseline note | Use **`$REPO/.venv-no-vllm`** (``setup_hf_parity_venv.sh``) for **`transformers==4.57.1`** / `hf_bench_match_jiuge.py`. vLLM + **Transformers 5** stays in **`.venv-vllm`** only. |

Reproduce the gate (must use a **real `.py` file**; vLLM workers use `spawn` and cannot start from `python -` / stdin):

```bash
REPO=/home/zenghua/workspace/minicpm5-moe-support
source "$REPO/.venv-vllm/bin/activate"
export PYTHONPATH="$REPO/InfiniLM/examples/vllm_patches:${PYTHONPATH:-}"
cd "$REPO/InfiniLM/examples"
export CUDA_VISIBLE_DEVICES=0   # optional
python vllm_probe_load.py --model-path /path/to/minicpm5 --enforce-eager
```

**Enable the MiniCPM5 MoE vLLM patch (recommended for this checkpoint):**

```bash
export PYTHONPATH="$REPO/InfiniLM/examples/vllm_patches:${PYTHONPATH:-}"
```

### Fallback model (vLLM path smoke test)

To confirm vLLM + GPU without MiniCPM, the same probe succeeds on a small local **Qwen3** checkpoint, e.g.:

```bash
REPO=/home/zenghua/workspace/minicpm5-moe-support
source "$REPO/.venv-vllm/bin/activate"
python "$REPO/InfiniLM/examples/vllm_probe_load.py" \
  --model-path /data-aisoft/mechdancer/models/Qwen3-0.6B \
  --max-model-len 256
```

**2026-04-14:** `LOAD_OK` on `Qwen3-0.6B` with `vllm==0.19.0`, A100, `CUDA_VISIBLE_DEVICES=0`.

---

## Environment (per run)

**Interpreter note (important):**

- **HF / InfiniLM** runs use **system** `python3` (with `.venv-vllm` **deactivated**).
- **vLLM** runs use **`$REPO/.venv-vllm`** (activated).

**Current container baseline (minicpm5-moe, 2026-04-14):** system `python3` imports
**`torch==2.10.0+cu128`** and **`transformers==4.57.1`**.
If your system `python3` differs (e.g. after a global `pip install vllm`), recreate the container or keep vLLM strictly inside `.venv-vllm`.

| Field | Value |
|--------|--------|
| Date | |
| Host / GPU | |
| CUDA / driver | |
| PyTorch | |
| Transformers | |
| vLLM | (if used; e.g. `0.19.0`; interpreter: `$REPO/.venv-vllm`) |
| Git commit / branch | |
| Model path | |

---

## Workload (keep identical across engines)

| Field | Typical value |
|--------|----------------|
| Prompt | `Hi` (or any fixed string) |
| Chat template | `apply_chat_template` + `add_generation_prompt=True` (same as `jiuge.py`) |
| Prompt token count | (recorded per run) |
| `max_new_tokens` |16 |
| Batch size | 1 |
| `top_k` / `top_p` / `temperature` | 1 / 1.0 / 1.0 |
| Activations dtype | bfloat16 |

---

## Metrics

### Weight load

| Engine | Metric | Notes |
|--------|--------|--------|
| InfiniLM | Wall time from loader start until ready to generate (`jiuge.py` “load weights over”) | Dominated by custom load path / I/O |
| Hugging Face | Wall time for `from_pretrained` + `.to(cuda)` + `eval()` (`hf_bench_match_jiuge.py`) | Different format and pipeline; not apples-to-apples with InfiniLM load |

### Generation (after weights are resident)

| Metric | Definition |
|--------|------------|
| **Total generation** | Wall time around the full generate path (InfiniLM: `jiuge` timer; HF: `model.generate` with CUDA sync) |
| **Prefill** | One forward over the full prompt (`use_cache=True`); HF bench reports this explicitly |
| **TTFT** | InfiniLM engine-reported time to first token (includes prefill-related work as implemented in engine) |
| **Decode total** | Sum of per-step decode forwards (HF manual loop) |
| **Decode avg / step** | `decode_total / max_new_tokens` |

---

## InfiniLM: MiniCPM5 MoE speed (fused vs reference MoE)

This compares **two MoE execution paths inside InfiniLM** for the **same** MiniCPM5-MoE checkpoint:

- **Fused**: `MiniCPM5MoeVllmFusedSparseMoeBlock` dispatches into `infinicore.vllm_fused_moe_bridge.fused_experts_ic` (vLLM `fused_experts`)
- **Reference**: set `INFINILM_DISABLE_VLLM_FUSED_MOE=1` to force the per-expert C++ loop (`MiniCPM5MoeSparseMoeBlock`)

### Flash-attn + paged KV, longer prompt (2026-04-21)

- **Prompt tokens**: 65 (as reported by `jiuge.py`)
- **max_new_tokens**: 128
- **CUDA_VISIBLE_DEVICES**: `1`
- **Attention**: `flash-attn` + `--enable-paged-attn` (paged KV)

| Path | Weight load (ms) | Total gen (ms) | TTFT (ms) | Prefill tok/s | Decode avg ITL (ms) | Decode tok/s |
|------|------------------:|---------------:|----------:|--------------:|---------------------:|-------------:|
| **Fused** (`INFINILM_FORCE_MOE_BACKEND=vllm_fused`) | 108722.62 | 4237.83 | 466.51 | 139.33 | 29.7 | 33.68 |
| **Reference** (`INFINILM_FORCE_MOE_BACKEND=baseline`) | 106274.83 | 9388.55 | 2835.14 | 22.93 | 51.6 | 19.38 |

### MoE: CPU router / pack vs fused kernel (microbench + e2e sweep)

**Vendor kernel vs PyPI vLLM (kernel-only, two processes):** same random tensors and sweep of ``num_tokens``; InfiniLM interpreter loads vendored ``torch.ops.infinilm`` path; **``.venv-vllm``** runs ``vllm.model_executor.layers.fused_moe.fused_experts`` (different torch ABI — never load ``_infinicore`` in the vLLM process). Align tuned JSON by pointing **both** runs at the same directory: ``--tuned-config-dir`` sets ``INFINILM_TUNED_CONFIG_FOLDER`` and ``VLLM_TUNED_CONFIG_FOLDER`` (copy ``E=...,N=...,device_name=....json`` from upstream ``vllm/model_executor/layers/fused_moe/configs/``; see ``InfiniCore/python/infinicore/vendor/vllm_fused_moe/NOTICE``).

```bash
REPO=/home/zenghua/workspace/minicpm5-moe-support
MODEL=/path/to/minicpm5   # optional; else pass explicit --num-experts --hidden --intermediate --top-k
SWEEP=1,8,32,128,512
TUNED=/path/to/copied_fused_moe_configs   # optional

export PYTHONPATH=$REPO/InfiniLM/python:$REPO/InfiniCore/python:${PYTHONPATH:-}
python3 $REPO/InfiniLM/examples/microbench_fused_moe_kernel.py --impl infinilm --nvidia \
  --model-path "$MODEL" --num-tokens-sweep "$SWEEP" --seed 0 --print-json \
  ${TUNED:+--tuned-config-dir "$TUNED"}

source "$REPO/.venv-vllm/bin/activate"
python $REPO/InfiniLM/examples/microbench_fused_moe_kernel.py --impl vllm --nvidia \
  --model-path "$MODEL" --num-tokens-sweep "$SWEEP" --seed 0 --print-json \
  ${TUNED:+--tuned-config-dir "$TUNED"}
```

Or: ``bash InfiniLM/examples/run_microbench_fused_moe_two_process.sh`` with ``REPO``, ``MODEL``, and optional ``TUNED`` / ``SWEEP`` / ``OUT`` / ``SEED``.

**Full-model prefill vs token count (baseline vs fused block):** each backend needs its **own subprocess** because ``INFINILM_FORCE_MOE_BACKEND`` is read when MoE layers are constructed. ``sweep_moe_e2e_backend_tokens.py`` wraps ``bench_balanced.py`` and records TTFT plus step breakdown (``ttft_gpu_forward_ms`` is whole forward on GPU, not MoE-only). For **router D2H + CPU top-k pack + H2D** vs **fused expert matmul**, use **Nsight Systems** on a fused run and look for NVTX: ``moe_vllm_fused::router_d2h_cpu_topk_pack_h2d`` vs ``moe_vllm_fused::fused_experts_dispatch`` inside ``MiniCPM5MoeVllmFusedSparseMoeBlock::forward`` (`InfiniLM/csrc/models/minicpm5_moe/minicpm5_moe_vllm_fused_sparse_moe_block.cpp`).

```bash
python3 $REPO/InfiniLM/examples/sweep_moe_e2e_backend_tokens.py --nvidia --model-path "$MODEL" \
  --prompt-tokens-sweep 64,256,1024 --max-new-tokens 1 --json-out /tmp/moe_sweep.json --print-summary

nsys profile -o moe_ns --trace-fork-before-exec=true \
  python3 $REPO/InfiniLM/examples/sweep_moe_e2e_backend_tokens.py --nvidia --model-path "$MODEL" \
  --backends vllm_fused --prompt-tokens-sweep 512 --json-out /tmp/moe_ns_dummy.json
```

**OpenAI-style concurrency:** larger effective batches raise per-step ``n_tokens``; compare ``InfiniLM/scripts/test_perf.py --concurrency`` (see ``minicpm5_moe_metrics_collection.md``) with the same MoE backend env; combine with Nsight if the server process is profiled from startup.

### vLLM (`vllm_bench_match_jiuge.py`) — aligned definitions

`LLM` is constructed with **`disable_log_stats=False`** so each finished `RequestOutput` carries **`RequestStateStats`**.

| Label | Source in vLLM | Match to jiuge / HF bench |
|--------|----------------|---------------------------|
| **TTFT** | `metrics.first_token_latency` (wall, arrival → first generated token) | Same role as jiuge first-step time for “first token out” |
| **Prefill (engine)** | `metrics.first_token_ts - metrics.scheduled_ts` (monotonic) | Roughly “in-model” prefill; not identical to HF’s isolated `forward` timing |
| **Decode total (engine)** | `metrics.last_token_ts - metrics.first_token_ts` (monotonic) | Spans all decode token steps |
| **Avg decode ITL / step** | `(last - first) / (num_generation_tokens - 1)` when `num_generation_tokens > 1` | Same exclusion of the first generated token as jiuge’s `time_measurements[1:]` mean |
| **Prefill tok/s** | `prompt_tokens / TTFT` | Same formula as jiuge printout when TTFT matches |
| **Decode tok/s** | `(n_generated - 1) / decode_engine_s` | Same as glossary under “decode excludes first output step” |
| **Load weights** | Wall time for `LLM(...)` construction | Includes engine init / compile / CUDA graphs unless `--enforce-eager` |

**Note:** Decode/prefill intervals use the engine’s **monotonic** timestamps; TTFT uses **wall** time. Throughput derived from monotonic decode length is comparable across runs on the same machine but not necessarily identical to wall-clock decode.

### Example snapshot (replace on every profile pass)

Numbers below are **examples only** from one session; do not treat them as baselines.

| Metric | InfiniLM (`jiuge.py`) | HF manual (prefill + greedy steps) | HF `model.generate()` |
|--------|----------------------|-------------------------------------|------------------------|
| Weight load | ~120 s | ~24 s | — |
| Total generation | ~1486 ms | — | ~5008 ms |
| Prefill | (see TTFT) | ~925 ms | (inside `generate`) |
| TTFT | ~669 ms | — | — |
| Decode total (16 steps) | — | ~4810 ms | — |
| Decode avg / step | ~54 ms | ~301 ms | — |
| Prefill + decode (manual) | — | ~5734 ms | — |

### vLLM snapshot (single prompt, MiniCPM5 MoE; patch enabled)

The numbers below come from `vllm_bench_match_jiuge.py` with `--enforce-eager` and the `sitecustomize` patch enabled.

| Engine | Prompt | Prompt tok | max_new_tokens | TTFT (ms) | Avg decode ITL (ms) | Prefill tok/s | Decode tok/s |
|--------|--------|------------|----------------|-----------|---------------------|---------------|--------------|
| vLLM (TransformersMoEForCausalLM) | `Hi` | 7 | 4 | ~591 | ~48.3 | ~11.84 | ~20.70 |
| vLLM (TransformersMoEForCausalLM) | A100 poem + MoE bullets | 38 | 16 | **149.73** | **39.49** | (see JSON) | (see JSON) |

---

## Reproduce

### InfiniLM

```bash
# Set REPO, PYTHONPATH, and linker env per your container / workspace rules.
python3 -u InfiniLM/examples/jiuge.py --nvidia \
  --model-path /path/to/minicpm5 \
  --prompt "Hi" --max-new-tokens 16 --batch-size 1 \
  --top-k 1 --top-p 1.0 --temperature 1.0 --attn default
```

**E2E rebuild + run (one line, inside `minicpm5-moe` container):**

```bash
docker exec minicpm5-moe bash -lc 'set -euo pipefail; export CUDA_VISIBLE_DEVICES=0; REPO=/home/zenghua/workspace/minicpm5-moe-support; MODEL=/data-aisoft/zenghua/models/minicpm5.16a3.v0314; PROMPT="Write a short poem about the A100 GPU, then explain in 3 bullet points what a mixture-of-experts model is and why routing matters."; export XMAKE_ROOT=y; export PYTHONPATH=$REPO/InfiniLM/python:$REPO/InfiniCore/python:${PYTHONPATH:-}; cd $REPO/InfiniCore; python3 scripts/install.py --nv-gpu=y --cuda_arch=sm_80 --aten=y; xmake build _infinicore; xmake install _infinicore; cd $REPO/InfiniLM; xmake build _infinilm; xmake install _infinilm; TORCH_LIB=$(python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), \"lib\"))"); FA=/usr/local/lib/python3.12/dist-packages/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so; (export LD_LIBRARY_PATH=/root/.infini/lib:$TORCH_LIB:/usr/local/lib/python3.12/dist-packages:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu; export LD_PRELOAD=$FA; cd $REPO/InfiniLM/examples; python3 -u jiuge.py --nvidia --model-path "$MODEL" --prompt "$PROMPT" --max-new-tokens 16 --batch-size 1 --top-k 1 --top-p 1.0 --temperature 1.0 --tp 1 --attn default)'
```

### Hugging Face (matched tokenization and sampling knobs)

```bash
python3 -u InfiniLM/examples/hf_bench_match_jiuge.py \
  --model-path /path/to/minicpm5 \
  --prompt "Hi" --max-new-tokens 16 --batch-size 1 \
  --top-k 1 --top-p 1.0 --temperature 1.0
```

### vLLM (jiuge-aligned `TokensPrompt`; single prompt)

Use the **vLLM venv** (see above). For MiniCPM5 MoE on vLLM 0.19, enable the patch and use `--enforce-eager`.

```bash
REPO=/home/zenghua/workspace/minicpm5-moe-support
source "$REPO/.venv-vllm/bin/activate"
export PYTHONPATH="$REPO/InfiniLM/examples/vllm_patches:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0   # optional

python -u "$REPO/InfiniLM/examples/vllm_bench_match_jiuge.py" \
  --model-path /path/to/model \
  --prompt "Hi" --max-new-tokens 16 --batch-size 1 \
  --top-k 1 --top-p 1.0 --temperature 1.0 \
  --max-model-len 8192

# Machine-readable:
python -u "$REPO/InfiniLM/examples/vllm_bench_match_jiuge.py" ... --json
```

For models that fault under `torch.compile`, add **`--enforce-eager`**.

---

## Continuous run log

| Run | Date | Engine | Load (s) | Prefill (ms) | Dec avg (ms) | Gen total (ms) | Notes |
|-----|------|--------|----------|--------------|--------------|----------------|-------|
| long-prompt-validate | 2026-04-14 | InfiniLM (`jiuge.py`) | — | — | **54.18** (avg ITL) | — | Prompt tok=39; TTFT=**1820.74** ms |
| long-prompt-validate | 2026-04-14 | HF (`hf_bench_match_jiuge.py`) | — | **1136.53** | **286.40** (avg/step) | — | Prompt tok=39; greedy |
| long-prompt-validate | 2026-04-14 | vLLM (`vllm_bench_match_jiuge.py`) | — | — | **39.49** (avg ITL) | — | Prompt tok=38; TTFT=**149.73** ms; `--enforce-eager` + patch |

---

## Related files

- `InfiniLM/examples/jiuge.py` — InfiniLM generation driver
- `InfiniLM/examples/hf_bench_match_jiuge.py` — HF timing with jiuge-aligned tokenization
- `InfiniLM/examples/setup_vllm_venv.sh` — create isolated `.venv-vllm` (`--moe` = Transformers 5 for MoE fallback)
- `InfiniLM/examples/vllm_probe_load.py` — minimal vLLM `LLM(...)` load probe (spawn-safe)
- `InfiniLM/examples/vllm_bench_match_jiuge.py` — single-prompt vLLM TTFT / decode ITL / throughput (jiuge-aligned tokens)
- `InfiniLM/examples/vllm_minicpm5_moe_patch.md` — why/how the MiniCPM5 MoE vLLM patch works
- `InfiniLM/examples/logit_sanity_minicpm5_moe.py` — correctness / logit sanity vs HF
