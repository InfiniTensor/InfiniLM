# Bounded chunked prefill

`prefill_chunk_size=0` (default) preserves the existing scheduling path. Set a positive value through `EngineConfig`, `LLM`, `AsyncLLMEngine`, or `--prefill-chunk-size` on the inference server / `examples/test_infer.py` to bound each prefill dispatch. For example:

```python
from infinilm.llm.llm import AsyncLLMEngine

engine = AsyncLLMEngine(
    model_path="/path/to/dense-text-model",
    tensor_parallel_size=2,
    cache_type="paged",
    enable_graph=False,
    prefill_chunk_size=512,
)
```

The effective prefill segment is at most `min(prefill_chunk_size, max_num_batched_tokens)` tokens. The supported parallel configurations are TP/PP=1/1, 2/1 and 1/2. PP=2 uses eager execution; TP=1/2 with PP=1 can combine eager Prefill and Decode graphs on CUDA-compatible devices. See [graph configuration](chunk-graphs.md). CLI parsing rejects other parallel sizes before starting a pipeline worker. Static cache, MLA, draft models, remote KV connectors, Mamba, MoE and multimodal inputs are excluded from this opt-in mode. TP=2 requires a working CUDA collective runtime (InfiniCCL built with CUDA/NCCL support); a CUDA model build alone does not establish that collectives are enabled. The direct-native examples `bench.py`, `llama.py` and `bench_videonsa.py` reject this option because they bypass the scheduler. Native validation used Qwen2.5-1.5B FP16 with paged attention on an A6000; this is not validation of every dense architecture/backend.

## Scheduling and ownership

Successful dispatches rotate among decode, prefill continuation, and new admission; empty or capacity-blocked phases are skipped. Each continuously eligible phase receives an opportunity within three successful dispatches. This is a bound on dispatch opportunities, not milliseconds or per-request admission latency. Decode batches retain `max_batch_size`; prefill dispatches contain one request. Continuations rotate FIFO. A capacity-deferred admission returns to the waiting queue.

Admission still reserves the full prompt's KV pages and accounts for the future decode capacity of partial requests. This change does not reduce prompt KV reservation. Existing prefix lookup pins only published full blocks; failed admission releases its temporary references. After successful model execution, only the completed span can be published. Cancellation returns all owned references, leaving computed full pages reusable. Intermediate chunks advance the computed boundary without emitting or appending sampled tokens; only the final chunk enters normal generation/EOS handling. Intermediate chunks use an internal `prefill_only` forward: the text causal-LM template skips the LM head, the worker skips sampling and token transfer, and the runner returns no sampled IDs. Final chunks and decode retain normal output handling.

This work follows the input-slicing/continuation concepts in [#371](https://github.com/InfiniTensor/InfiniLM/pull/371), adapted to the target branch's per-request KV publication. At audited revision `02419ee3`, #371's executable phase order favors waiting, then continuation, then decode (with periodic forced continuation), so sustained admissions can defer decode. The new mode adds explicit phase rotation, per-request publication and cancellation handling. [#571](https://github.com/InfiniTensor/InfiniLM/pull/571), audited at `6683db7e`, orders waiting admissions with priority/aging and addresses a separate concern. This is not a claim of first implementing chunked prefill, nor an integration of those two PRs.

## Validation and tradeoffs

Run native-free regressions with Python 3.10+ and `janus`, `xxhash` installed:

```sh
python -m unittest discover -s test/llm -p 'test_chunk_*.py'
```

The chunk tests cover configuration forwarding/exclusions, disabled/static paths, offsets including non-page-aligned chunks, shared prefixes, publication boundaries, EOS/final output, queued and in-flight cancellation, generation hashing, capacity/pin rollback and phase progress.

One short A6000 run used eager Qwen2.5-1.5B FP16, 256-token pages, 128 pages, batch limit 2, prefix caching off, and three requests: an active 128-token prompt generating 128 tokens, an arriving 8192-token prompt generating 16, and a late 128-token prompt generating 16. With 512-token chunks:

| Measurement | Disabled | Chunk 512 |
|---|---:|---:|
| Active request maximum output gap | 1804.46 ms | 227.92 ms |
| Active request p95 output gap | 25.64 ms | 136.68 ms |
| Late short request first token | 1781.56 ms | 77.08 ms |
| Long request first token | 1790.27 ms | 2098.04 ms |
| All three requests finish | 2.928 s | 3.015 s |

All 160 generated token IDs matched. These are single-run delivery timestamps, not stable throughput or universal speedup estimates. Breaking one long stall into several smaller stalls improves the maximum gap while worsening p95 in this trace. The late request starts earlier but also encounters gaps while the long prefill continues.

A separate native lifecycle check used 300-token chunks across 256-token pages, a 1027-token prompt, cancellation after the first chunk, reuse of its 256-token published prefix, and repeat reuse of a 1024-token prefix. Ownership, publication bounds, no intermediate output and completion checks passed. **Strict greedy sequence equality failed** between the partial-prefix/chunk path and full-prefix reuse at generated index 6. At identical prompt/prior tokens, the competing FP16 logits were tied at 16.015625 in the first path, versus 16.03125 and 16.015625 on reuse. Full-prefix repeats were stable, and a normal-forward rerun matched the diagnostic outputs. This is evidence of sensitivity to small numerical differences across execution shapes, not a proof of general numerical equivalence. Preserve this failed strict comparison when evaluating the feature; no model-quality benchmark is claimed.


## TP=2 execution contract

The scheduler owns one request state and one logical page table. `InferEngine::forward` fans out the same input to both rank workers; each worker copies positions, sequence lengths, offsets, block tables and slots to its own device and sets its thread-local attention metadata. Dense tensor parallelism partitions attention heads while retaining the token coordinates. The existing segmented processor inputs therefore express the same prefill span on both ranks; the TP=2 extension changes the configuration guards and adds native validation, without a new native slicing implementation.

Engine-owned KV writes, reads and subsequent reuse are ordered on persistent device streams. Worker `wait()` is not an all-device completion barrier: rank zero synchronizes after forward (after sampling on the ordinary output path), while another rank can report completion after submission. The supported serialized engine path preserves stream ordering for later reuse. Cross-stream KV transfer and overlap are outside this contract. No additional global device synchronization is inserted in the production path.

Run explicit native checks with an FP16 dense model and a working CUDA/NCCL runtime:

```sh
CUDA_VISIBLE_DEVICES=0,1 python test/llm/check_chunk_tp.py --model /path/to/model --tp 2 --chunk-size 300 --output /tmp/chunk-tp2.json
CUDA_VISIBLE_DEVICES=0,1 python test/llm/benchmark_chunk_prefill.py --model /path/to/model --tp 2 --chunk-size 0 --output /tmp/mixed-off.json
CUDA_VISIBLE_DEVICES=0,1 python test/llm/benchmark_chunk_prefill.py --model /path/to/model --tp 2 --chunk-size 512 --output /tmp/mixed-512.json
```

The correctness script watches the first and last local attention layers on each rank/stage. Before every prefill it fills scheduled K/V slots with NaNs, then verifies all scheduled values are finite and values outside the span are unchanged. These probes synchronize/copy KV only in this script; the benchmark does not use them. TP=2 chunk 300, TP=1 chunk 300, TP=2 chunk disabled and TP=2 chunk 300 with prefix caching disabled all passed. They cover non-page-aligned spans, partial and full prefix reuse, shared references, cancellation and full capacity recovery. The same completed prompts generated identical tokens across these four fixtures. Controlled cancellation after forward covers a final segment with caching enabled and an intermediate segment with caching disabled; it is not a network cancellation race test.

## TP=2 short mixed-request measurements

Two A6000s, Qwen2.5-1.5B FP16, eager paged attention, the same three-request workload above, prefix caching disabled. Each configuration ran three times; entries are medians of per-run metrics. Disabled/512 runs alternated order; 1024 ran subsequently, so shared-machine drift is not fully controlled.

| Measurement | Disabled | Chunk 512 | Chunk 1024 |
|---|---:|---:|---:|
| Active maximum output gap | 1022.94 ms | 142.74 ms | 239.21 ms |
| Active p95 output gap | 24.88 ms | 92.68 ms | 63.62 ms |
| Late short request first token | 1001.54 ms | 68.16 ms | 139.44 ms |
| Long request first token | 1012.17 ms | 1435.76 ms | 1210.33 ms |
| Finite-window output throughput | 75.27 token/s | 62.83 token/s | 68.06 token/s |

Chunk 512 reduced maximum output gap by 86.0% and late TTFT by 93.2%, with 41.8% longer long-request TTFT and 16.5% lower output throughput. Chunk 1024 reduced those gaps by 76.6% and 86.1%, with 19.6% longer long TTFT and 9.6% lower throughput. Both worsened active p95: one long stall becomes multiple shorter stalls. Throughput is 160 delivered output tokens divided by elapsed time from earliest request submission to last token; loading and warmup are excluded. These short shared-machine measurements do not establish sustained serving throughput or an optimal chunk size.

All nine TP=2 runs matched the same 160 baseline token IDs. Traces confirm 16 or 8 long-prefill segments, active decode progress between segments and late first-token delivery before the long prefill finished. Prefer 512 when reducing the largest pauses matters most; 1024 is a candidate for a smaller throughput penalty. The default remains zero. Prefill/decode are separate dispatches, not a fused mixed batch, and the original measurements in this table predate the output-suppression optimization below.

The prior strict numerical failure was also reproduced under TP=2 and remains **strict_mismatch**. At generated index 6 with identical prior tokens, the partial-prefix path selected token 220 with logit 16.03125 versus token 13 at 16.015625; full-prefix reuse tied both at 16.015625 and selected 13. Two full-prefix repeats agreed. This supports a narrow FP16 execution-shape sensitivity diagnosis; it does not prove universal output equivalence or model-quality preservation. Review this opt-in extension with that limitation visible.


## Avoid unused intermediate outputs

The internal `prefill_only` flag is enabled automatically only for non-final chunks. It defaults to false in native inputs and is propagated to every TP rank. The shared text causal-LM template computes the model/KV state and omits last-token selection and the vocabulary projection. The rank worker omits GPU sampling, token D2H transfer and retained output tensors. The Python forward returns `None`; the runner returns an empty sampled-ID list, which the existing intermediate-chunk lifecycle handles before normal output-count validation. Other model implementations can still compute logits internally; only the shared text template's LM-head bypass was implemented here.

Rank zero still synchronizes its device stream before returning. The optimization does not relax publication/cancellation completion or introduce cross-stream overlap. To preserve the previous stochastic sampling progression, the worker still makes one identical `uniform_real_distribution<float>` RNG draw per request and discards it. This is a source-level equivalence argument; greedy tests do not establish stochastic output equality. Native guards reject all-position sampling and missing request offsets. For PP, all stages receive the same intermediate flag, finish their activation transfers, and skip the sampled-token exchange. The coordinator waits for each stage before publishing or reclaiming KV pages. Intermediate chunks run eagerly unless the opt-in fixed-size MetaX Prefill graph matches.

**Rebuild the InfiniLM native extension when using this change.** InfiniCore does not need a source change. Additional checks:

```sh
python -m unittest discover -s test/llm -p 'test_chunk_*.py'
CUDA_VISIBLE_DEVICES=0,1 python test/llm/check_chunk_output.py --model /path/to/model --tp 2 --chunk-size 300 --output /tmp/chunk-output.json
CUDA_VISIBLE_DEVICES=0,1 python test/llm/benchmark_chunk_prefill.py --model /path/to/model --tp 2 --chunk-size 512 --legacy-chunk-output --output /tmp/legacy-output.json
CUDA_VISIBLE_DEVICES=0,1 python test/llm/benchmark_chunk_prefill.py --model /path/to/model --tp 2 --chunk-size 512 --output /tmp/skip-output.json
```

`--legacy-chunk-output` is a benchmark control: it restores intermediate logits, sampling, transfer and conversion using the same compiled binary and unchanged schedule. It is not an engine policy option. The CPU tests include intermediate/final/decode forwarding and cancellation with an empty output. Native TP1/TP2 and prefix-off checks confirm absent intermediate raw outputs, valid final outputs and unchanged KV lifecycle; invalid native calls are rejected without preventing subsequent valid inference.

Three alternating TP2 chunk512 comparisons on the same short workload gave these per-run medians:

| Measurement | Legacy output work | Skip intermediate output |
|---|---:|---:|
| Total long-prefill dispatch execution | 1202.99 ms | 1175.69 ms |
| Intermediate long-prefill dispatch execution | 1077.84 ms | 1051.54 ms |
| Long request TTFT | 1437.11 ms | 1426.41 ms |
| Active maximum output gap | 141.38 ms | 140.92 ms |
| Active p95 output gap | 93.97 ms | 92.33 ms |
| Late short request TTFT | 75.52 ms | 75.26 ms |
| Finite-window output throughput | 64.19 token/s | 62.87 token/s |

Measured long-prefill execution decreased 2.3%, while overall throughput did **not** improve in the median (-2.0%). Per-run throughput ranged 61.05–68.14 versus 61.17–67.19 token/s, with changes in both directions between paired runs. These measurements support removing redundant work, not a stable end-to-end throughput gain or recovery of the earlier chunking penalty. All six runs matched the same 160 baseline tokens, retained 16 long-prefill segments and let active decode and late admission progress. The preserved near-tied FP16 fixture still reports `strict_mismatch`; this optimization does not resolve that numerical limitation.


A separate same-input CUDA/NVTX replay confirmed that across 15 intermediate chunks on two ranks, the optimization removed 30 LM-head GEMVs, 30 last-token selections, 45 sampling-related kernels and 15 token D2H copies. Paged-prefill attention and NCCL invocation counts stayed at 840 and 1680. They accounted for approximately 73% and 12% of summed kernel duration across both GPUs in the legacy replay; those sums are not wall-clock shares. The trace supports targeting paged-prefill attention efficiency and batch scheduling for further throughput work, rather than expecting the small sampling cost to explain the entire chunking penalty. Replay/instrumentation timings are excluded from the serving comparison.

## Integrated cache and parallel validation

`prefix_cache_policy="lru"` is the default; `"slru"` enables bounded probation/protected queues. Chunk admission records prefix hits only after allocation succeeds, so capacity-deferred requests cannot promote cache entries. The same page manager handles eager and graph execution.

On two A6000s, Qwen2.5-1.5B FP16 passed 13 combinations covering single-device/TP2 eager and Decode graphs, PP2 eager, LRU/SLRU and prefix caching disabled. Completed shared/repeated requests matched one baseline token sequence, first/last local KV layers passed poisoned-slot checks, and all references returned to zero. PP worker checks separately cover the second stage. This finite set does not remove the FP16 numerical limitation documented above.

Build InfiniCore with `--graph=y` for device graphs and CUDA/NCCL collective support for TP/PP. A build without device graphs can replay recorded operators on the host; an enabled LM flag alone does not demonstrate CUDA graph execution. The native test can check actual launches with an optional counter:

```sh
g++ -std=c++17 -shared -fPIC -I"$INFINI_ROOT/include" test/llm/graph_counter.cc -ldl -o /tmp/infini-graph-counter.so
LD_PRELOAD=/tmp/infini-graph-counter.so CUDA_VISIBLE_DEVICES=0,1 python test/llm/check_chunk_tp.py --model /path/to/fp16-model --tp 2 --chunk-size 300 --graph --policy slru --output /tmp/tp2-graph.json
```

For PP, start the same checker twice using separate `CUDA_VISIBLE_DEVICES` values, `--tp 1 --pp 2`, a shared `--port`, and `--stage 1` on the worker. Both processes need their own `--output` file. PP graphs and combined TP2/PP2 are excluded.
