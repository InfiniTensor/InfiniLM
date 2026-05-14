# Vendor vs upstream MoE — concurrency-focused delta memo

This memo supports **localizing overhead when client concurrency is greater than one** between `INFINILM_MOE_FUSED_STACK=vendor` and `upstream`. Items are tagged **(hypothesis)** until backed by a trace or JSON from the profiling checklist in [`minicpm5_moe_inference_profiling.md`](minicpm5_moe_inference_profiling.md) (see subsection **Concurrency (c>1) vendor vs upstream**).

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

**Filled table (session 2026-05-14, `CUDA_VISIBLE_DEVICES=1`, A100, synthetic E=32,H=768,N=384,top_k=4; two torch builds — treat `wall_ms_per_iter` as primary cross-impl sanity):**

| num_tokens | vendor cuda_ms_mean | upstream cuda_ms_mean | Δ% vs vendor (cuda) |
|------------|--------------------:|------------------------:|--------------------:|
| 8 | 3.080 | 0.261 | −91.5% |
| 32 | 3.490 | 0.250 | −92.8% |
| 128 | 3.485 | 0.251 | −92.8% |
| 512 | 5.794 | 0.291 | −95.0% |

**Observation (confirmed for microbench only):** vendor fused-experts kernel time is **flat ~3.5 ms** for 8–128 tokens then rises at 512; upstream stays **~0.25–0.29 ms** on this harness — large gap is **kernel / dispatch**, not OpenAI concurrency. Interpret cross-interpreter `cuda_ms_mean` cautiously (see profiling doc caveats).

### E2E concurrency golden matrix (exploratory cross-stack)

Artifact: `bench_artifacts/e2e_moe_fused_stack_concurrency_summary.json` from [`run_moe_fused_stack_concurrency_sweep.sh`](run_moe_fused_stack_concurrency_sweep.sh) with `CONCURRENCIES="1 8"`, `NUM_REQUESTS=8`, `max_tokens=4`, prompt `Hi` (vendor: flash+paged; upstream: default attn+static).

| c | Δ RPS (up−vendor) | Δ avg_ttft_s | Δ avg_decode_ms_per_chunk | Δ avg_latency_s |
|---|------------------:|-------------:|--------------------------:|----------------:|
| 1 | +0.016 | −0.0075 | +1.24 | −0.0038 |
| 8 | −0.54 | +0.47 | −439.33 | −0.84 |

**Caution:** The large **negative** Δ on `avg_decode_ms_per_chunk` at c=8 mixes stacks (attn/KV/streaming); chunk wall semantics can differ between legs — pair with Nsight (checklist §5) before attributing to MoE alone.

---

## Prioritized vendor backlog (max 5; revise after traces)

Ground in NVTX / nsys / microbench before coding.

**Evidence update (2026-05-14, synthetic microbench token sweep):** vendor `cuda_ms_mean` is **~3×–20×** slower than upstream `fused_experts` at the same shapes across `num_tokens` 8–512 — prioritize **fused experts / Triton / tuned configs** before router-only work, unless Nsight shows router dominates in full-model forwards.

1. **Tune / port configs** — Point `INFINILM_TUNED_CONFIG_FOLDER` at vLLM-style JSON for `(E, N, device)`; re-run token sweep; if gap collapses at large B, prioritize tuning over kernel rewrites.
2. **Reduce bridge sync** — Profile `vllm_fused_moe_dispatch.cpp` + Python bridge under batched forwards; remove redundant `cudaStreamSynchronize` / GIL scope if proven hot on c>1 paths.
3. **Port high-value upstream `fused_moe.py` hunks** — Use the reproducible diff in profiling doc; cherry-pick dispatch / workspace reuse changes since vendor pin.
4. **Router fast path** — If router microbench + profiler shows `topk`/`masked_fill` bound at MoE B seen in production, consider `INFINILM_USE_VLLM_GROUPED_TOPK_KERNEL=1` on vendor where shapes allow, or tighten vendor Torch graph (fewer temporaries).
5. **Scheduler instrumentation** — If microbench is flat but e2e worsens with c, add lightweight counters (batch size, queue wait) before touching Triton.
