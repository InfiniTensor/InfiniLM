# MiniCPM5 MoE — inference profiling (InfiniLM vs HF vs vLLM)

Use this document as a **continuous profiling** log: fill the environment block each session, then append or update the run table with fresh numbers. Workload settings should stay fixed when comparing engines.

**Structured metrics log:** for single-prompt smokes (**InfiniLM `bench_balanced.py` + vLLM**; HF optional elsewhere) and e2e OpenAI server runs, use **`minicpm5_moe_metrics_collection.md`** (tables + suggested JSON paths under `bench_artifacts/`).

### MoE fused stack — primary knob (`INFINILM_MOE_FUSED_STACK`)

| Value | Router | Fused experts | Typical interpreter |
|--------|--------|----------------|---------------------|
| **`vendor`** (default when unset) | `torch.ops.infinilm.minicpm5_grouped_sigmoid_topk` (vendored reference; optional `INFINILM_USE_VLLM_GROUPED_TOPK_KERNEL` applies on this stack) | `infinilm::outplace_fused_experts` via ATen dispatcher when enabled, else `infinicore.vllm_fused_moe_bridge.fused_experts_ic` → **vendored** `infinicore.vendor.vllm_fused_moe` Triton | Container **system `python3`** (current e2e default) |
| **`vendor_router_cpu`** | D2H → CPU `run_router_topk_cpu` → pack → H2D in `MiniCPM5MoeVllmFusedSparseMoeBlock` (same vendored experts as **`vendor`**) | Same as **`vendor`** | Same as **`vendor`** |
| **`upstream`** | `vllm.model_executor.layers.fused_moe.router.grouped_topk_router.grouped_topk` | **`vllm.model_executor.layers.fused_moe.fused_moe.fused_experts`** (PyPI / site-packages, not the in-repo vendor copy) | **`$REPO/.venv-vllm/bin/python`** (isolated torch + vLLM 0.19.x) |

When `INFINILM_MOE_FUSED_STACK` is **unset**, the effective stack is **`vendor`**. Setting **`INFINILM_USE_VLLM_GROUPED_TOPK_KERNEL=1`** on the vendor stack prints a **one-line stderr reminder** (once per process) to prefer explicit `INFINILM_MOE_FUSED_STACK` for stack selection. **`INFINILM_MOE_ROUTER`** / **`INFINILM_MOE_ROUTER_ENGINE`** are **ignored** (stderr notice once if set); use **`vendor_router_cpu`** instead of the old `INFINILM_MOE_ROUTER=cpu`.

InfiniLM’s reference router `run_router_topk_cpu` in `InfiniLM/csrc/models/minicpm5_moe/minicpm5_moe_router_cpu_detail.hpp` uses per-row `nth_element`-style partial top-k on CPU, while the vLLM-aligned GPU path uses `torch.topk` (optionally with `sorted=True` when batch invariance is required upstream). **Tie-breaking and expert ordering can differ** when logits tie; end-to-end text can still match if downstream is insensitive.

**Escapes / diagnostics:**

- **`INFINILM_FORCE_MOE_BACKEND=baseline`**: reference per-expert MoE (construction-time).
- **`INFINILM_DISABLE_VLLM_FUSED_MOE=1`**: per-forward fallback to the reference MoE block inside the fused class.
- **`INFINILM_VLLM_FUSED_DISPATCH=legacy`**: skip the ATen boxed dispatcher for experts; force the Python `fused_experts_ic` path (still honors `INFINILM_MOE_FUSED_STACK` for vendor vs upstream experts).

**Finer vendor-router controls (`vendor` / `vendor_router_cpu` stack, GPU router branch only):**

- **`INFINILM_MOE_ROUTER_SORTED_TOPK=1`**: `sorted=True` on internal `torch.topk` in the vendored grouped-sigmoid reference.
- **`INFINILM_USE_VLLM_GROUPED_TOPK_KERNEL=1`**: try `vllm._custom_ops.grouped_topk` on CUDA when shapes allow; **skipped** when `INFINILM_MOE_FUSED_STACK=upstream` (upstream enters the vLLM `grouped_topk` path first).

**Tolerance for numeric tests:** compare `topk_weights` with `rtol=1e-4`, `atol=1e-5` (float32 matmul pipeline); compare `topk_ids` exactly after sorting columns when checking sets, or exact match when the reference uses the same `sorted=` flag.

**NVTX (Nsight):** with **`vendor`**, GPU router: `moe_vllm_fused::router_grouped_topk_gpu`; fused experts: `moe_vllm_fused::fused_experts_dispatch`. With **`vendor_router_cpu`**, the CPU-router envelope is `moe_vllm_fused::router_d2h_cpu_topk_pack_h2d`. With **`upstream`**: `moe_vllm_fused::router_vllm_upstream` and `moe_vllm_fused::fused_experts_vllm_upstream`. `moe_vllm_fused::router_d2h_cpu_topk_pack_h2d` also wraps fallback when the GPU router bridge returns nullopt.

### Vendor ↔ upstream file map (vLLM 0.19.x) and reproducible diffs

Use this table when reconciling InfiniLM’s **vendored** tree with **installed** vLLM under `.venv-vllm` (paths are relative to the repo root vs `site-packages`).

| Concern | Vendor path (this repo) | Upstream reference (venv) | Notes (`InfiniCore/python/infinicore/vendor/vllm_fused_moe/NOTICE`) |
|--------|-------------------------|---------------------------|----------------------------------------------------------------------|
| Grouped sigmoid router op | `InfiniCore/python/infinicore/vendor/vllm_fused_moe/minicpm5_grouped_sigmoid_topk.py` | `vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py` | Vendor adds `torch.ops.infinilm.minicpm5_grouped_sigmoid_topk`, `INFINILM_MOE_FUSED_STACK`, optional `_custom_ops.grouped_topk` |
| Fused experts / Triton | `InfiniCore/python/infinicore/vendor/vllm_fused_moe/fused_moe.py` | `vllm/model_executor/layers/fused_moe/fused_moe.py` | Vendor registers `torch.ops.infinilm.*`; SwiGLU / platform shims per NOTICE |
| Torch op registration glue | `InfiniCore/python/infinicore/vendor/vllm_fused_moe/torch_register.py` | `vllm/utils/torch_utils.py` (`direct_register_custom_op`) | Different library handle (`infinilm` vs `vllm`) |

**Reproducible diff procedure (run inside `minicpm5-moe`)**

Resolve `site-packages` from `import vllm` so the same command works across Python minor versions in the image.

**One-shot: export upstream MoE sources, diff vs vendor, print metrics**

```bash
docker exec minicpm5-moe bash -lc 'set -euo pipefail
REPO=/home/zenghua/workspace/minicpm5-moe-support
VPY="$REPO/.venv-vllm/bin/python"
SP="$($VPY -c "import vllm, os; print(os.path.dirname(vllm.__file__))")"
OUT="$REPO/third_party/vllm-0.19.x-site-export"
mkdir -p "$OUT/model_executor/layers/fused_moe/router"
cp -a "$SP/model_executor/layers/fused_moe/fused_moe.py" "$OUT/model_executor/layers/fused_moe/"
cp -a "$SP/model_executor/layers/fused_moe/router/grouped_topk_router.py" "$OUT/model_executor/layers/fused_moe/router/"

V_FUSED="$REPO/InfiniCore/python/infinicore/vendor/vllm_fused_moe/fused_moe.py"
V_ROUTER="$REPO/InfiniCore/python/infinicore/vendor/vllm_fused_moe/minicpm5_grouped_sigmoid_topk.py"
U_FUSED="$OUT/model_executor/layers/fused_moe/fused_moe.py"
U_GRP="$OUT/model_executor/layers/fused_moe/router/grouped_topk_router.py"

echo "=== stack ==="
$VPY -c "import vllm, torch; print(\"vllm\", vllm.__version__); print(\"torch\", torch.__version__)"

echo "=== wc -l (upstream_export vs vendor) ==="
wc -l "$U_FUSED" "$V_FUSED" "$U_GRP" "$V_ROUTER"

echo "=== sha256 upstream export (repro fingerprint) ==="
sha256sum "$U_FUSED" "$U_GRP"

echo "=== unified diff line counts (full -u output) ==="
diff -u "$U_FUSED" "$V_FUSED" | wc -l
diff -u "$U_GRP" "$V_ROUTER" | wc -l
'
```

Inspect hunks interactively (still in the container):

```bash
docker exec -it minicpm5-moe bash -lc 'REPO=/home/zenghua/workspace/minicpm5-moe-support; OUT="$REPO/third_party/vllm-0.19.x-site-export"; diff -u "$OUT/model_executor/layers/fused_moe/fused_moe.py" "$REPO/InfiniCore/python/infinicore/vendor/vllm_fused_moe/fused_moe.py" | less'
```

(`grouped_topk_router.py` is not a line-for-line peer of `minicpm5_grouped_sigmoid_topk.py`; use the **fused_moe** diff for experts, and the **router** diff only as a coarse cross-check against vLLM’s `grouped_topk` call path.)

**Example metrics — source diff (captured in-container, 2026-05-14)**

| Metric | Value |
|--------|--------|
| `vllm` | `0.19.0` |
| `torch` (`.venv-vllm`) | `2.10.0+cu128` |
| Lines: upstream `fused_moe.py` (export) | 2312 |
| Lines: vendor `fused_moe.py` | 1859 |
| Lines: upstream `grouped_topk_router.py` (export) | 350 |
| Lines: vendor `minicpm5_grouped_sigmoid_topk.py` | 208 |
| Unified diff lines (`-u` export vs vendor `fused_moe.py`) | 594 |
| Unified diff lines (`-u` grouped_topk_router vs minicpm5_grouped_sigmoid_topk`) | 526 |
| `sha256` export `fused_moe.py` | `607c0a459306a71ff7d01445772494367f3924098739bbd3b4f43020738297d4` |
| `sha256` export `grouped_topk_router.py` | `8dd610aed59432cdb0e5ef6eec280cc70af74c1f791ee35be1e3ba4af730e97a` |

Re-run the one-shot after upgrading the venv; refresh this table when the numbers change.

### Vendor ↔ upstream **overhead gap** (fused experts microbench, container)

Besides **source diff** metrics above, reproduce a **timing gap** for the same MoE tensor contract as `INFINILM_MOE_FUSED_STACK`:

- **Vendor experts path analog:** `microbench_fused_moe_kernel.py --impl infinilm` (system `python3`, vendored Triton `fused_experts`).
- **Upstream experts path analog:** `microbench_fused_moe_kernel.py --impl vllm` (`.venv-vllm`, PyPI `vllm...fused_experts`).

Script: [`run_moe_fused_stack_microbench_gap.sh`](InfiniLM/examples/run_moe_fused_stack_microbench_gap.sh) — sets `PYTHONPATH`, normalizes `LD_LIBRARY_PATH` for system `python3` + `_infinicore`, runs both impls with the **same** `--seed` / shapes, then prints **CUDA event mean** and **wall ms / iter**, and writes `bench_artifacts/microbench_moe_fused_stack_gap.json`.

**Default shapes:** synthetic `E=32, H=768, N=384, top_k=4`, `num_tokens=128` (fits busy GPUs). For **real checkpoint** `(E, N, H, top_k)` from `config.json`, use a mostly-free device and:

```bash
docker exec minicpm5-moe bash -lc 'export CUDA_VISIBLE_DEVICES=1; export MICROBENCH_GAP_USE_MODEL=1; bash /home/zenghua/workspace/minicpm5-moe-support/InfiniLM/examples/run_moe_fused_stack_microbench_gap.sh'
```

**Synthetic microbench (one-liner, in-container)**

```bash
docker exec minicpm5-moe bash -lc 'export CUDA_VISIBLE_DEVICES=1; bash /home/zenghua/workspace/minicpm5-moe-support/InfiniLM/examples/run_moe_fused_stack_microbench_gap.sh'
```

Pick `CUDA_VISIBLE_DEVICES` from `nvidia-smi` (container) so the process has headroom; **GPU 0 is often full** on shared hosts.

**Caveats (read before interpreting the gap):**

- This is **fused experts only** (no InfiniLM router, no full decode stack). End-to-end overhead is tracked separately (e.g. §2.1.3 e2e JSON, `bench_balanced.py`).
- **Two different Torch builds** (image NV stack vs `.venv-vllm` `+cu128`); `cuda_ms_mean` from CUDA events is only weakly comparable across builds — treat **`wall_ms_per_iter`** as the primary cross-interpreter sanity check when both runs succeed.
- Both sides log **missing tuned fused_moe JSON** for `(E=32, N=384)` on A100; copying upstream `configs/*.json` into `INFINILM_TUNED_CONFIG_FOLDER` narrows kernel choice noise when chasing small gaps.

**Example overhead metrics (captured in-container, 2026-05-14, `CUDA_VISIBLE_DEVICES=1`, synthetic shapes, default warmup/iters)**

| Metric | Vendor (`infinilm`) | Upstream (`vllm`) | Δ (upstream − vendor) |
|--------|--------------------:|------------------:|------------------------:|
| `cuda_ms_mean` | 3.5315 | 0.2545 | −3.2770 ms (−92.8%) |
| `wall_ms_per_iter` | 3.5632 | 0.2836 | −3.2796 ms (−92.0%) |
| `torch` | `2.10.0a0+b4e4ee81d3.nv25.12` | `2.10.0+cu128` | (not matched) |

On this run, **upstream is faster** on the synthetic microbench (negative Δ). Your gap sign/magnitude will change with `MICROBENCH_GAP_USE_MODEL`, tuned configs, and GPU load — re-run the script and replace the table.

**Review discipline:** keep exports **local** (or commit `third_party/vllm-0.19.x-site-export` only if your team wants a frozen pin in git) and paste relevant `diff -u` hunks into this log for the session.

### Concurrency (c>1) vendor vs upstream

**Goal:** separate **OpenAI client concurrency** effects from **kernel-only** gaps. Code-review map and hypotheses: [`MOE_VENDOR_UPSTREAM_CONCURRENCY_MEMO.md`](MOE_VENDOR_UPSTREAM_CONCURRENCY_MEMO.md).

**Golden scripts (same `test_perf` shape per row; concurrency varies):**

- [`run_moe_fused_stack_concurrency_sweep.sh`](InfiniLM/examples/run_moe_fused_stack_concurrency_sweep.sh) — boots **vendor** (system `python3`, flash-attn, paged) then **upstream** (`.venv-vllm`, default attn, static); writes `bench_artifacts/e2e_moe_stack_vendor_concurrency_c{N}_hi.json`, `e2e_moe_stack_upstream_concurrency_c{N}_hi.json`, and `e2e_moe_fused_stack_concurrency_summary.json`.
- [`run_moe_fused_stack_microbench_token_sweep.sh`](InfiniLM/examples/run_moe_fused_stack_microbench_token_sweep.sh) — sweeps `num_tokens` for **fused experts only** (no server); writes `bench_artifacts/microbench_moe_fused_stack_gap_by_num_tokens.json`.

**One-shot (container, pick a free GPU index):**

```bash
docker exec minicpm5-moe bash -lc 'export CUDA_VISIBLE_DEVICES=1; bash /home/zenghua/workspace/minicpm5-moe-support/InfiniLM/examples/run_moe_fused_stack_microbench_token_sweep.sh'
docker exec minicpm5-moe bash -lc 'export CUDA_VISIBLE_DEVICES=1; bash /home/zenghua/workspace/minicpm5-moe-support/InfiniLM/examples/run_moe_fused_stack_concurrency_sweep.sh'
```

Default concurrency list is `1 2 4 8` (long: two full model loads). For a shorter run: `export CONCURRENCIES="1 8"`.

**Mixed-stack caveat:** vendor vs upstream e2e legs differ in **Python, attention, and KV cache**; cross-stack JSON deltas are **exploratory**. Same-stack **c curves** (vendor-only or upstream-only) isolate scheduler + streaming under load.

**E2E JSON column template (fill from `e2e_moe_fused_stack_concurrency_summary.json`):**

| `concurrency` | stack | `requests_per_second` | `avg_ttft_s` | `avg_decode_ms_per_chunk` | `avg_latency_s` |
|---------------|--------|----------------------:|-------------:|---------------------------:|----------------:|
| 1 | vendor | (from JSON) | … | … | … |
| 1 | upstream | … | … | … | … |
| … | … | … | … | … | … |

**Profiling checklist (cheap → expensive)** — use to shrink **vendor** overhead; each line: *run* → *target* → *signal*.

1. **Golden commands** — Save `docker exec` lines + `bench_artifacts/*.json` for reproducibility. *Signal:* stable curves vs `c`.
2. **Router micro-equivalence** — [`InfiniCore/test/test_minicpm5_grouped_sigmoid_topk.py`](InfiniCore/test/test_minicpm5_grouped_sigmoid_topk.py) / tolerances in § MoE fused stack above. *Signal:* outputs match ⇒ focus on sync/compute, not logits math.
3. **`torch.profiler`** — One forward or `microbench_fused_moe_kernel.py` hot path with **large `num_tokens`**. *Signal:* `%` in `topk`, `masked_fill`, `copy_`, small CUDA kernels.
4. **Token sweep** — `run_moe_fused_stack_microbench_token_sweep.sh`. *Signal:* vendor time grows with B while microbench flat under c>1 e2e ⇒ triage **scheduler**, not Triton.
5. **Nsight Systems** — NVTX: `moe_vllm_fused::router_grouped_topk_gpu`, `moe_vllm_fused::fused_experts_dispatch` (and upstream ranges if applicable); capture **c=1** vs **c=8** client load. *Signal:* idle/sync growth between ranges when `c` rises.
6. **Nsight Compute (selective)** — One vendor `fused_experts` kernel at representative B. *Signal:* DRAM vs math bound.
7. **py-spy** — Server PID during `c=8` `test_perf`. *Signal:* GIL / asyncio vs GPU-bound stacks.
8. **Stretch: stack parity** — Same venv + attn + `cache_type` for both stacks (e.g. flash in `.venv-vllm`) so MoE is isolated. *Signal:* if gap vanishes, prior e2e delta was **environmental**.

**Prioritized vendor backlog (max 5):** see memo § “Prioritized vendor backlog”; refresh after steps 1–8.

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
  --model-path "$MODEL" --num-tokens-sweep "$SWEEP" --seed 0 --print-json -v \
  ${TUNED:+--tuned-config-dir "$TUNED"}

source "$REPO/.venv-vllm/bin/activate"
python $REPO/InfiniLM/examples/microbench_fused_moe_kernel.py --impl vllm --nvidia \
  --model-path "$MODEL" --num-tokens-sweep "$SWEEP" --seed 0 --print-json -v \
  ${TUNED:+--tuned-config-dir "$TUNED"}
```

Or: ``bash InfiniLM/examples/run_microbench_fused_moe_two_process.sh`` with ``REPO``, ``MODEL``, and optional ``TUNED`` / ``SWEEP`` / ``OUT`` / ``SEED``.

**Full-model prefill vs token count (baseline vs fused block):** each backend needs its **own subprocess** because ``INFINILM_FORCE_MOE_BACKEND`` is read when MoE layers are constructed. ``sweep_moe_e2e_backend_tokens.py`` wraps ``bench_balanced.py`` and records TTFT plus step breakdown (``ttft_gpu_forward_ms`` is whole forward on GPU, not MoE-only). Pass ``-v`` / ``--verbose`` to log each subprocess start/end and stream ``bench_balanced`` timing prints (stderr). For **router D2H + CPU top-k pack + H2D** vs **fused expert matmul**, use **Nsight Systems** on a fused run and look for NVTX: ``moe_vllm_fused::router_d2h_cpu_topk_pack_h2d`` vs ``moe_vllm_fused::fused_experts_dispatch`` or ``moe_vllm_fused::fused_experts_vllm_upstream`` (stack-dependent) inside ``MiniCPM5MoeVllmFusedSparseMoeBlock::forward`` (`InfiniLM/csrc/models/minicpm5_moe/minicpm5_moe_vllm_fused_sparse_moe_block.cpp`).

```bash
python3 $REPO/InfiniLM/examples/sweep_moe_e2e_backend_tokens.py --nvidia --model-path "$MODEL" \
  --prompt-tokens-sweep 64,256,1024 --max-new-tokens 1 --json-out /tmp/moe_sweep.json --print-summary -v

nsys profile -o moe_ns --trace-fork-before-exec=true \
  python3 $REPO/InfiniLM/examples/sweep_moe_e2e_backend_tokens.py --nvidia --model-path "$MODEL" \
  --backends vllm_fused --prompt-tokens-sweep 512 --json-out /tmp/moe_ns_dummy.json -v
```

### Nsight Systems — **vendor** vs **upstream** MoE fused stack (when e2e gap warrants)

Use this after [`run_e2e_moe_fused_stack_compare.sh`](InfiniLM/examples/run_e2e_moe_fused_stack_compare.sh) (or §2.1.3 in ``minicpm5_moe_metrics_collection.md``) shows a **persistent** decode-proxy or latency gap you want to explain beyond known attn/KV/torch differences. Capture **two** reports (same GPU index, quiet machine): one server boot with ``INFINILM_MOE_FUSED_STACK=vendor``, one with ``upstream``.

**NVTX ranges** (timeline markers inside ``MiniCPM5MoeVllmFusedSparseMoeBlock::forward``): **vendor** GPU router ``moe_vllm_fused::router_grouped_topk_gpu`` + experts ``moe_vllm_fused::fused_experts_dispatch``; **upstream** ``moe_vllm_fused::router_vllm_upstream`` + ``moe_vllm_fused::fused_experts_vllm_upstream``; CPU-router envelope ``moe_vllm_fused::router_d2h_cpu_topk_pack_h2d`` for ``INFINILM_MOE_FUSED_STACK=vendor_router_cpu`` or when the GPU router bridge returns nullopt.

**Hygiene:** ``unset LD_LIBRARY_PATH`` then set a **single** torch-aligned prefix (see workspace docker perf rule); **do not** ``LD_PRELOAD`` image flash-attn while running ``.venv-vllm`` Python. For kernel ordering debug only, ``CUDA_LAUNCH_BLOCKING=1`` is acceptable; remove for representative timelines.

**Artifacts directory (example):**

```bash
REPO=/home/zenghua/workspace/minicpm5-moe-support
MODEL=/data-aisoft/zenghua/models/minicpm5.16a3.v0314
NSDIR="$REPO/InfiniLM/examples/bench_artifacts/nsys_moe_stack"
mkdir -p "$NSDIR"
```

**1) Vendor stack — profile server from process start (port 8016 example)**

```bash
docker exec minicpm5-moe bash -lc 'set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
REPO=/home/zenghua/workspace/minicpm5-moe-support
MODEL=/data-aisoft/zenghua/models/minicpm5.16a3.v0314
NSDIR="$REPO/InfiniLM/examples/bench_artifacts/nsys_moe_stack"
mkdir -p "$NSDIR"
export PYTHONPATH=$REPO/InfiniLM/python:$REPO/InfiniCore/python:${PYTHONPATH:-}
TORCH_LIB=$(python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), \"lib\"))")
unset LD_LIBRARY_PATH
unset LD_PRELOAD
export LD_LIBRARY_PATH=/root/.infini/lib:$TORCH_LIB:/usr/local/lib/python3.12/dist-packages:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu
export INFINILM_FORCE_MOE_BACKEND=vllm_fused
export INFINILM_MOE_FUSED_STACK=vendor
cd "$REPO/InfiniLM/python"
nsys profile -t cuda,nvtx,osrt -o "$NSDIR/moe_vendor_server" \
  python3 -m infinilm.server.inference_server --nvidia \
    --model_path "$MODEL" --dtype bfloat16 --attn flash-attn --cache_type paged \
    --num_blocks 256 --block_size 256 --max_batch_size 8 --max_tokens 256 \
    --port 8016 --host 127.0.0.1
'
```

In a **second** shell, drive load while the profiled server runs (``test_perf.py`` or a short fixed client), then stop the server with Ctrl+C so ``nsys`` finalizes ``moe_vendor_server.nsys-rep``. Optionally add Nsight’s CUDA capture-range flags if you instrument ``cudaProfilerStart/Stop`` around the workload.

**2) Upstream stack — profile ``.venv-vllm`` server (port 8017 example)**

```bash
docker exec minicpm5-moe bash -lc 'set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
REPO=/home/zenghua/workspace/minicpm5-moe-support
MODEL=/data-aisoft/zenghua/models/minicpm5.16a3.v0314
VPY="$REPO/.venv-vllm/bin/python"
NSDIR="$REPO/InfiniLM/examples/bench_artifacts/nsys_moe_stack"
mkdir -p "$NSDIR"
export PYTHONPATH=$REPO/InfiniLM/python:$REPO/InfiniCore/python:${PYTHONPATH:-}
export INFINILM_FORCE_MOE_BACKEND=vllm_fused
export INFINILM_MOE_FUSED_STACK=upstream
TORCH_LIB="$("$VPY" -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), \"lib\"))")"
unset LD_LIBRARY_PATH
unset LD_PRELOAD
export LD_LIBRARY_PATH=/root/.infini/lib:$TORCH_LIB:/usr/local/lib/python3.12/dist-packages:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu
cd "$REPO/InfiniLM/python"
nsys profile -t cuda,nvtx,osrt -o "$NSDIR/moe_upstream_server" \
  "$VPY" -m infinilm.server.inference_server --nvidia \
    --model_path "$MODEL" --dtype bfloat16 --attn default --cache_type static \
    --num_blocks 256 --block_size 256 --max_batch_size 8 --max_tokens 256 \
    --port 8017 --host 127.0.0.1
'
```

**3) In-proc short slice (optional):** wrap ``jiuge.py`` or ``bench_balanced.py`` the same way if the server timeline is too noisy; keep MoE env vars identical to the stack under test.

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
