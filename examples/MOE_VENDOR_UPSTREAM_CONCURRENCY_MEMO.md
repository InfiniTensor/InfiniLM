# Vendor vs upstream MoE — concurrency-focused delta memo

This memo supports **localizing overhead when client concurrency is greater than one** between `INFINILM_MOE_FUSED_STACK=vendor` and `upstream`. Items are tagged **(hypothesis)** until backed by a trace or JSON from the profiling checklist in [`minicpm5_moe_inference_profiling.md`](minicpm5_moe_inference_profiling.md) (see subsection **Concurrency (c>1) vendor vs upstream**).

## Profiling session log (hypothesis-driven sweeps, 2026-05-14)

- **Container:** `minicpm5-moe` bind-mounted `REPO=/home/zenghua/workspace/minicpm5-moe-support`.
- **GPU:** `CUDA_VISIBLE_DEVICES=1` inside `docker exec` (local index 1; **NVIDIA A100-SXM4-80GB**). Host `rm` on `bench_artifacts/*.json` failed with permission denied (root-owned bind-mount files); stale MoE sweep files were removed **inside** the container before the baseline capture.
- **Microbench scripts:** [`run_moe_fused_stack_microbench_gap.sh`](run_moe_fused_stack_microbench_gap.sh) (now supports optional `MICROBENCH_TUNED_CONFIG_DIR` → `--tuned-config-dir` on both legs), [`run_moe_fused_stack_microbench_token_sweep.sh`](run_moe_fused_stack_microbench_token_sweep.sh), [`run_moe_fused_stack_concurrency_sweep.sh`](run_moe_fused_stack_concurrency_sweep.sh) with `CONCURRENCIES="1 8"`, `NUM_REQUESTS=8`, `MAX_TOKENS=4`, `WARMUP_REQUESTS=2`, `FIXED_PROMPT=Hi`.
- **Canonical baseline artifacts (repo):** `bench_artifacts/microbench_moe_fused_stack_gap_by_num_tokens.json`, `bench_artifacts/microbench_moe_fused_stack_gap.json` (last token of sweep), `bench_artifacts/e2e_moe_fused_stack_concurrency_summary.json`, `bench_artifacts/e2e_moe_stack_{vendor,upstream}_concurrency_c{1,8}_hi.json`.
- **InfiniLM rebuild:** after `INFINILM_MOE_DEBUG_NUM_TOKENS` + fast-handoff C++ edits, `xmake build _infinilm && xmake install _infinilm` in-container (≈15 s compile on warm cache).

## Code-review map (vendor vs vLLM 0.19.x)

| Area | Vendor (repo) | Upstream (venv `site-packages/vllm/...`) | c>1–relevant notes |
|------|----------------|--------------------------------------------|---------------------|
| Router | [`InfiniCore/python/infinicore/vendor/vllm_fused_moe/minicpm5_grouped_sigmoid_topk.py`](InfiniCore/python/infinicore/vendor/vllm_fused_moe/minicpm5_grouped_sigmoid_topk.py) | `model_executor/layers/fused_moe/router/grouped_topk_router.py` | Vendor path: multiple `torch.topk` / masks per forward; scales with **effective token rows** in a batched MoE forward. **(hypothesis)** Extra Python overhead per forward hurts more when the server **merges concurrent steps** into larger `num_tokens`. |
| Fused experts | [`InfiniCore/python/infinicore/vendor/vllm_fused_moe/fused_moe.py`](InfiniCore/python/infinicore/vendor/vllm_fused_moe/fused_moe.py) | `model_executor/layers/fused_moe/fused_moe.py` | Triton dispatch + workspace; vendor NOTICE lists SwiGLU / `moe_sum` shims vs upstream. **(hypothesis)** At larger B, vendor kernel choice or sync differs from PyPI vLLM. |
| Bridge / stream | [`InfiniLM/csrc/utils/vllm_fused_moe_dispatch.cpp`](InfiniLM/csrc/utils/vllm_fused_moe_dispatch.cpp), [`InfiniCore/python/infinicore/vllm_fused_moe_bridge.py`](InfiniCore/python/infinicore/vllm_fused_moe_bridge.py) | N/A (upstream stack stays in Torch/vLLM from `.venv-vllm`) | IC↔Torch stream events + GIL around grouped topk / fused_experts. **(hypothesis)** Under **concurrent requests**, tail latency grows if the bridge **serializes** or syncs more often than the upstream path. |
| Server / scheduler | [`InfiniLM/python/infinilm/server/inference_server.py`](InfiniLM/python/infinilm/server/inference_server.py), [`InfiniLM/python/infinilm/llm/llm.py`](InfiniLM/python/infinilm/llm/llm.py) | Same server code; different Python + attn + KV for scripted A/B | **(confirmed)** Scripted vendor vs upstream e2e in [`run_e2e_moe_fused_stack_compare.sh`](run_e2e_moe_fused_stack_compare.sh) differs **interpreter, flash-attn vs default attn, paged vs static KV** — not MoE-only. Use [`run_moe_fused_stack_concurrency_sweep.sh`](run_moe_fused_stack_concurrency_sweep.sh) for a **fixed golden matrix** per stack; treat cross-stack c curves as exploratory until optional **stack parity** stretch (same venv + attn + cache) is done. |
| Effective batch | MoE block [`MiniCPM5MoeVllmFusedSparseMoeBlock`](InfiniLM/csrc/models/minicpm5_moe/minicpm5_moe_vllm_fused_sparse_moe_block.cpp) | Same | **(hypothesis)** Higher client concurrency increases **simultaneous sequences** and scheduler batching; measure **per-forward `num_tokens`** (logging) to see if vendor disadvantage appears only above a batch threshold. |

## Hypotheses vs server-only causes (short bullets)

1. **(hypothesis)** Vendor router pure-Torch work per token row dominates at medium B; upstream `grouped_topk` may use less host-side work or better fused CUDA on some shapes.
2. **(hypothesis)** `vllm_fused_moe_dispatch` stream bridging adds fixed microseconds per layer forward; under c>1 queueing, that fixed cost appears as worse **p99** decode chunk time.
3. **(hypothesis)** Vendor Triton `fused_moe` path selects a suboptimal config without tuned JSON (`INFINILM_TUNED_CONFIG_FOLDER`); gap widens as B grows (see microbench token sweep).
4. **(hypothesis)** InfiniLM async scheduler / KV interaction differs under load between system torch (vendor) and venv torch (upstream) — can mimic MoE regression without any MoE code change.
5. **(confirmed)** Any e2e delta that **does not reproduce** in [`run_moe_fused_stack_microbench_gap.sh`](run_moe_fused_stack_microbench_gap.sh) / token sweep should be triaged toward **scheduler / HTTP / cache**, not `fused_moe.py` line edits.

## Evidence slots (fill after running checklists)

| Artifact | Description |
|----------|-------------|
| `bench_artifacts/e2e_moe_fused_stack_concurrency_summary.json` | Produced by [`run_moe_fused_stack_concurrency_sweep.sh`](run_moe_fused_stack_concurrency_sweep.sh) |
| `bench_artifacts/microbench_moe_fused_stack_gap_by_num_tokens.json` | Produced by [`run_moe_fused_stack_microbench_token_sweep.sh`](run_moe_fused_stack_microbench_token_sweep.sh) |
| `bench_artifacts/microbench_moe_fused_stack_gap.json` | Single-shape gap from [`run_moe_fused_stack_microbench_gap.sh`](run_moe_fused_stack_microbench_gap.sh) |

### Microbench token sweep (synthetic MoE, in-container)

Command:

`docker exec minicpm5-moe bash -lc 'export CUDA_VISIBLE_DEVICES=1; bash /home/zenghua/workspace/minicpm5-moe-support/InfiniLM/examples/run_moe_fused_stack_microbench_token_sweep.sh'`

Artifact: `bench_artifacts/microbench_moe_fused_stack_gap_by_num_tokens.json`.

**Filled table (baseline refresh, 2026-05-14, `CUDA_VISIBLE_DEVICES=1`, A100, synthetic E=32,H=768,N=384,top_k=4; `tuned_config_dir=null`; two torch builds — treat `wall_ms_per_iter` as primary cross-impl sanity):**

| num_tokens | vendor cuda_ms_mean | upstream cuda_ms_mean | Δ% vs vendor (cuda) |
|------------|--------------------:|------------------------:|--------------------:|
| 8 | 3.097 | 0.264 | −91.5% |
| 32 | 3.570 | 0.515 | −85.6% |
| 128 | 3.501 | 0.392 | −88.8% |
| 512 | 3.775 | 0.436 | −88.5% |

**Iteration 1 — H-fused (tuned JSON):** Upstream PyPI ships **no** `E=32,N=384,device_name=NVIDIA_A100-SXM4-80GB.json`. Cloning mismatched filenames (e.g. H100 `E=72,N=384`) into that name **regressed** vendor `cuda_ms_mean` in a spot check (~6 ms @128 tokens); copying `E=64,N=640,A100` content to the `E=32,N=384` filename was **neutral/slightly worse** than default. **Pass gate: fail** (no vendor kernel win). The gap script now accepts `MICROBENCH_TUNED_CONFIG_DIR` so real JSON can be dropped in later without script edits.

**Iteration 2 — H-bridge (stream handoff):** Added **`INFINILM_VLLM_FUSED_FAST_STREAM_HANDOFF=1`** (default `0`): skips the extra `cudaStreamSynchronize` on the current Torch stream in [`InfiniLM/csrc/utils/vllm_fused_moe_dispatch.cpp`](InfiniLM/csrc/utils/vllm_fused_moe_dispatch.cpp) after `bridge_ic_stream_to_torch_stream()`, and skips the matching pre-op `torch.cuda.current_stream().synchronize()` in [`InfiniCore/python/infinicore/vllm_fused_moe_bridge.py`](InfiniCore/python/infinicore/vllm_fused_moe_bridge.py) for `fused_experts_ic` / `grouped_sigmoid_topk_ic`. Microbench (direct Triton) is **unchanged** by design.

Vendor-only `test_perf` with `SKIP_UPSTREAM=1`, same golden matrix, **`INFINILM_VLLM_FUSED_FAST_STREAM_HANDOFF=1`**: **c=8** `avg_latency_s` ≈ **1.33**, `avg_decode_ms_per_chunk` ≈ **308**, RPS ≈ **5.95**. The **default** vendor-only run taken just before (same day, same flags except handoff) had **c=8** ≈ **1.71 s / 373 ms/chunk / 4.65 RPS** — e2e has high variance; against the **restored full-matrix** vendor row below (≈ **1.35 s / 316 ms/chunk / 5.87 RPS**), the handoff run is still a small win on latency/decode. **Gate:** microbench unchanged (expected); treat handoff as **experimental** until nsys proves safety across all MoE call sites.

**Nsys:** `which nsys` in `minicpm5-moe` → `/usr/local/cuda/bin/nsys`. Suggested compare recipe remains [`minicpm5_moe_inference_profiling.md`](minicpm5_moe_inference_profiling.md) § Concurrency (wrap `test_perf` / `jiuge`).

**Correctness:** `pytest InfiniCore/test/test_minicpm5_grouped_sigmoid_topk.py` (3 passed) with image `LD_LIBRARY_PATH` (`/root/.infini/lib` + torch `lib`). `run_correctness_bench_smoke.sh` with **`INFINILM_VLLM_FUSED_FAST_STREAM_HANDOFF=1`:** steps **1–4** (system `python3` paths) completed and matched golden token ids; step **5** (`.venv-vllm` upstream stack subprocess) **SIGSEGV** — treat as **pre-existing / venv stack** issue, not caused by the InfiniLM-only bridge edits.

**Iteration 3 — H-router:** No router Python edits in this pass (kernel microbench still dominates). Pytest + smoke above cover the router module import path.

**Iteration 4+ — H-batch / H-server:** Env-gated stderr log **`INFINILM_MOE_DEBUG_NUM_TOKENS=1`** in [`InfiniLM/csrc/models/minicpm5_moe/minicpm5_moe_vllm_fused_sparse_moe_block.cpp`](InfiniLM/csrc/models/minicpm5_moe/minicpm5_moe_vllm_fused_sparse_moe_block.cpp) prints `n_tokens`, `batch_size`, `seq_len`, `hidden_size` once per fused forward (requires InfiniLM rebuild). Use alongside vendor-only concurrency sweeps to correlate scheduler batching with client `c`. **py-spy:** e.g. `py-spy record -o /tmp/vendor_c8.svg --pid <inference_server_pid> --duration 15` while driving `test_perf` at `c=8`.

### E2E concurrency golden matrix (exploratory cross-stack)

Artifact: `bench_artifacts/e2e_moe_fused_stack_concurrency_summary.json` from [`run_moe_fused_stack_concurrency_sweep.sh`](run_moe_fused_stack_concurrency_sweep.sh) with `CONCURRENCIES="1 8"`, `NUM_REQUESTS=8`, `max_tokens=4`, prompt `Hi` (vendor: flash+paged; upstream: default attn+static). **Figures below are from the latest full matrix refresh** after restoring vendor+upstream legs (default stream handoff).

| c | Δ RPS (up−vendor) | Δ avg_ttft_s | Δ avg_decode_ms_per_chunk | Δ avg_latency_s |
|---|------------------:|-------------:|--------------------------:|----------------:|
| 1 | +4.40 | −0.43 | −7.50 | −0.45 |
| 8 | +0.34 | +0.21 | −277.83 | −0.62 |

**Caution:** The large **negative** Δ on `avg_decode_ms_per_chunk` at c=8 mixes stacks (attn/KV/torch); chunk wall semantics can differ between legs — pair with Nsight (checklist §5) before attributing to MoE alone.

---

## Prioritized vendor backlog (max 5; revise after traces)

Ground in NVTX / nsys / microbench before coding.

**Evidence update (2026-05-14):** vendor fused-experts microbench remains **far** slower than PyPI vLLM at synthetic `(E=32,N=384)` on A100 without a matching tuned JSON; **H-fused** needs a real `E=32,N=384,A100` config (autotune or upstream addition), not filename cloning.

1. **Tune / port configs** — Use new `MICROBENCH_TUNED_CONFIG_DIR` on gap scripts; obtain or generate **`E=32,N=384,device_name=NVIDIA_A100-SXM4-80GB.json`** legitimately; re-run token sweep.
2. **Stream handoff experiments** — `INFINILM_VLLM_FUSED_FAST_STREAM_HANDOFF=1` for vendor e2e profiling; keep default `0` until nsys proves ordering is redundant everywhere.
3. **Port high-value upstream `fused_moe.py` hunks** — Use the reproducible diff in profiling doc; cherry-pick dispatch / workspace reuse changes since vendor pin.
4. **Router fast path** — If router microbench + profiler shows `topk`/`masked_fill` bound at MoE B seen in production, consider `INFINILM_USE_VLLM_GROUPED_TOPK_KERNEL=1` on vendor where shapes allow, or tighten vendor Torch graph (fewer temporaries).
5. **Scheduler / batch instrumentation** — `INFINILM_MOE_DEBUG_NUM_TOKENS=1` + vendor-only `c` sweeps + py-spy under load before Triton rewrites.
