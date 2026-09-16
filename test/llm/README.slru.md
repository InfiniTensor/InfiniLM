# Optional segmented LRU

SLRU protects prefixes reused by admitted requests against one-off traffic.
The default policy remains `lru`. Enable SLRU for a paged-cache engine:

```python
from infinilm.llm import AsyncLLMEngine

engine = AsyncLLMEngine(
    model_path="/path/to/model",
    cache_type="paged",
    prefix_cache_policy="slru",
    prefix_cache_protected_ratio=0.8,
)
```

The server exposes the same options:

```bash
python python/infinilm/server/inference_server.py --model /path/to/model \
  --enable-paged-attn --prefix-cache-policy slru \
  --prefix-cache-protected-ratio 0.8
```

`examples/test_infer.py` forwards the same options when using
`--enable-paged-attn`. The offline `examples/bench.py` workflow explicitly disables
prefix caching and does not measure these eviction policies.

The ratio must be strictly between zero and one. SLRU with static cache is
rejected. Disabling prefix caching prevents both reuse and promotion.

## Policy semantics

- Newly published blocks become probationary when their final reference is
  released. Unpublished blocks immediately return to the free pool.
- A locally matched prefix is promoted after successful request admission.
  Lookup alone, token-budget deferral, rejected admission, failed allocation,
  and publishing newly computed tokens do not promote blocks. A request
  canceled after admission still counts as admitted reuse.
- Only zero-reference blocks enter either reclaim queue. Protected membership
  survives pinning, including references held by remote KV transfers.
- Protected membership is limited to `floor(num_blocks * ratio)`, including
  pinned blocks. This is a cap, not reserved GPU memory. A one-block pool has
  no protected capacity.
- Successful reuse and final release refresh protected ordering. Exceeding the
  cap demotes the oldest protected member to probationary without releasing
  references. A demoted pinned block becomes evictable only on final release.
- Reclamation consumes probationary blocks first, then protected blocks if
  required. Protection never prevents an otherwise possible allocation.
- Hit promotion and release process a request's block table tail first. This
  favors retaining its earlier prefix pages, but does not implement a global
  radix-tree leaf constraint.

The implementation adds ordered protected membership and a protected reclaim
queue. Candidate selection and individual queue updates use expected O(1)
metadata operations; promoting a prefix costs O(number of matched blocks),
including at most that many demotions. No history of evicted hashes is retained.

## Framework precedent and limits

[SGLang's SLRU documentation at revision 4b186cf](https://github.com/sgl-project/sglang/blob/4b186cfea59371cc8ec38597f750a5597f7147a0/docs/docs/advanced_features/radix_eviction_policy.mdx)
describes probationary/protected priorities based on hit counts. Its eligible
victims are unlocked radix-tree leaves. This implementation instead uses
physical pages, admitted-request reuse, and an explicit protected capacity cap;
it is not a port or a claim of equivalent behavior.

SLRU is workload dependent. A large stale protected set can slow adaptation to
new hotspots. Pure cyclic scans larger than the cache can still miss on every
request because no resident page receives a second hit. Large requests may
require evicting protected blocks too. Choose the ratio from representative
traffic rather than assuming the default is optimal.

## Verification

```bash
python -m unittest discover -s test/llm -p 'test_*.py'
```

Coverage includes hotspot retention versus LRU, admission-only promotion,
bounded protection, pinned demotion, tail preference, physical-ID reuse,
mixed shared lifetimes, and scheduler rejection/cancellation/deferred transfer
paths. Configuration tests exercise the CLI, server startup configuration,
engine constructors, and paged scheduler forwarding without loading a model.

A small real-scheduler metadata trace with an eight-block pool and ratio 0.5
kept both two-page hotspots after six one-off prefixes. Across ten measurement
requests, cached tokens rose from 64 to 128 and prefill fell from 266 to 202
(16-token blocks). No-reuse and cyclic traces produced zero hits for both
policies. These are metadata observations, not GPU throughput measurements.

A bounded RTX A6000 / Qwen2.5-1.5B FP16 check used 256-token blocks, ratio 0.5,
four warmup requests, and ten measurement requests with eight generated tokens
each. At eight blocks, LRU recorded 1,024 cached tokens and 4,416 prefill tokens;
SLRU recorded 2,048 and 3,392 respectively (23.19% less prefill). All 14 output
token sequences and the separate normal-EOS check matched across LRU/8 blocks,
SLRU/8 blocks, LRU/32 blocks, and SLRU/8 blocks with prefix caching disabled.
These short single-run checks establish bounded correctness and retention
behavior, not a stable throughput gain or general workload coverage.
