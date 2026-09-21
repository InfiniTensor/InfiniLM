# Paged prefix caching and chunked Prefill

Paged prefix caching uses LRU by default. Chunking is disabled by default.
For a dense text model, optional SLRU and bounded Prefill can be configured
through `EngineConfig`, `LLM`, or `AsyncLLMEngine`:

```python
from infinilm.llm import AsyncLLMEngine

engine = AsyncLLMEngine(
    model_path="/path/to/model",
    cache_type="paged",
    prefix_cache_policy="slru",
    prefix_cache_protected_ratio=0.8,
    prefill_chunk_size=512,
    tensor_parallel_size=1,
    enable_graph=False,
)
```

The inference server and `examples/test_infer.py` expose the same settings:

```sh
python python/infinilm/server/inference_server.py --model /path/to/model \
  --enable-paged-attn --prefix-cache-policy slru \
  --prefix-cache-protected-ratio 0.8 --prefill-chunk-size 512
```

## Cache policies

Only zero-reference pages can be reclaimed. Final release processes a request's
pages tail first, favoring its earlier prefix. Shared or remote-transfer owners
keep their pages pinned until the last reference is released.

SLRU reclaims probationary pages before protected pages. A resident prefix is
promoted only after successful admission; lookup, rejected admission and newly
computed KV do not promote it. Protected membership includes pinned pages and
is capped at `floor(num_blocks * prefix_cache_protected_ratio)`. The oldest
protected member is demoted when this cap is exceeded. The ratio must be
strictly between zero and one; this cap does not reserve GPU memory.

SLRU requires paged cache. Disabling prefix caching disables reuse and promotion.
It retains no history of evicted hashes. It can preserve established hotspots
through one-use traffic, but stale protected pages can delay adaptation to new
hotspots, and overcapacity cyclic traffic can still miss on every request.

## Scheduling and parallel execution

A Prefill dispatch computes at most
`min(prefill_chunk_size, max_num_batched_tokens)` tokens from one request.
Successful dispatches rotate among Decode, Prefill continuation and admission;
empty or blocked phases are skipped. Decode reuses the ordinary scheduler and
its batch limit. This bounds dispatch opportunities, not elapsed latency.

Admission still reserves all prompt pages and future Decode headroom. Prefix
lookup pins only published full pages; failed admission releases temporary pins.
After successful execution, only completed full pages are published. Cancellation
releases request ownership while preserving reusable completed pages.
Intermediate chunks execute every Transformer layer but omit the LM head,
sampling and output tokens. The final chunk enters normal generation handling.

Supported chunk configurations are TP/PP = 1/1, 2/1 and 1/2. TP2 requires working
CUDA collectives; PP2 uses eager execution and publishes only after both stages
complete. Static cache, MLA, draft models, remote KV connectors, Mamba, MoE and
multimodal inputs are excluded from chunking. The direct-native `bench.py`,
`llama.py` and `bench_videonsa.py` examples reject chunking because they bypass
the scheduler; `bench.py` also disables prefix caching.

## Device graphs

With `device="cuda"` and `enable_graph=True`, TP1/TP2 with PP1 can combine eager
Prefill and Decode graphs. InfiniCore must be built with `--graph=y`. Tested
attention backends are `paged-attn` on NVIDIA A6000 and `flash-attn` on MetaX
C500; `cuda` maps to MACA on the MetaX build. Other devices have not been
validated for these combinations.

Intermediate Prefill chunks use eager execution. A single-token final tail
may use the existing Decode graph because it has the same one-query attention
semantics. This change adds no Prefill graph capture or configuration switch.
PP2 chunking remains eager.

## Validation and tradeoffs

See [test instructions](../test/llm/README.md) for CPU regressions and opt-in
native lifecycle checks. Hardware validation covered A6000 Qwen2.5-1.5B FP16
and C500 Qwen3-0.6B/4B BF16. This does not establish support for every dense
architecture or backend. Quantized-model chunking has not been validated;
the dense-model check does not reject quantization metadata.

Chunking can reduce long output pauses and short-request waiting while reducing
throughput and increasing long-request TTFT. Graphs add initialization time and
retained workspace; full memory overhead and sustained HTTP throughput were
not measured. A historical FP16 near-tied-token mismatch across prefix execution
shapes remains documented; universal bitwise equivalence is not promised.

[PR #573](https://github.com/InfiniTensor/InfiniLM/pull/573) contains the measured
benefits, costs, test conditions and links to archived reports and experiments.
