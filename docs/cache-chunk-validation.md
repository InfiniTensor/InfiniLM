# Cache, chunking and graph integration validation

The default is LRU eviction with chunking disabled. SLRU and chunking are
optional. Prefix cache hits are promoted only after successful admission in
both scheduling paths. Intermediate chunks execute all Transformer layers;
the LM head and sampling are omitted until the final chunk.

## Tested configurations (2026-09-16)

| Hardware / model | Execution | Coverage |
|---|---|---|
| A6000, Qwen2.5-1.5B FP16 | TP1 and TP2, eager or eager Prefill + Decode graphs | LRU, SLRU, non-page-aligned chunk300, prefix reuse, shared ownership, cancellation, prefix caching off |
| Two A6000s, same model | TP1/PP2 eager | Unchunked baseline, chunk300 + LRU/SLRU, prefix caching off; both stages inspected |
| C500 slice, Qwen3-0.6B BF16 | TP1, Decode graph with eager or graph Prefill | LRU/SLRU, chunk512, prefix reuse, cancellation, prefix caching off |
| C500 slice, Qwen3-4B BF16 | TP1, Prefill and Decode graphs | SLRU, chunk512, shared/repeated prefix and cancellation |

All 13 A6000 configurations matched one completed-request token reference.
The six C500 lifecycle configurations passed; 0.6B outputs matched across
policies/modes and each model's shared/repeated requests matched. First/last
local KV layers were checked by poisoning scheduled slots, requiring finite
writes there and unchanged values elsewhere. Every case returned all 16
blocks to a usable state with zero references. Actual device-graph launches
were counted; TP2 Decode replay launched a graph on each rank.

The matching InfiniCore MetaX varlen ABI fix also passed six Prefill and six
Decode FP16/BF16 operator cases against FP32 reference attention. Its build
selects the signature of the installed wheel; older signatures have fixture
coverage, not execution on older hardware/software installations.

Structured results: [cache-chunk-graphs.json](validation/cache-chunk-graphs.json).
The CPU suite now has 111 passing tests, including successful-admission-only
SLRU promotion, PP worker configuration forwarding, shared Decode scheduling
across page boundaries and connector metadata on idle dispatch:

```sh
python -m unittest discover -s test/llm -p 'test_*.py'
```

Native reproduction commands and prerequisites are in
[chunked-prefill.md](chunked-prefill.md) and [chunk-graphs.md](chunk-graphs.md).
`check_chunk_tp.py` accepts `--tp`, `--pp`, `--stage`, `--port`, `--graph`,
`--policy`, `--cache-off`, `--chunk-size` and `--output`. Use separate processes
and visible devices for PP stages. The counter is required only with `--graph`.

## Scheduler reuse follow-up

Both Prefill policies now reuse the original Decode scheduling path; output
construction also shares speculative operations and connector metadata setup.
The configuration tests share one isolated module loader. The follow-up passed
111 CPU tests (108 existing and three additional cases) and one A6000 smoke
run: Qwen2.5-1.5B FP16, TP1/PP1, SLRU, page256/pool16, chunk300, eager Prefill
and Decode graphs. It observed 14 device-graph launches, matched the archived
same-configuration output tokens, and returned all 16 pages with zero references
after shared/repeated requests and cancellation. It used the refactored Python
source with the unchanged native binary from the integration checks. This was
a correctness check; the full GPU matrix and archived performance measurements
were not repeated for this Python refactor.

## Performance boundaries

The [component performance report](performance-ablation.md) provides baseline
versus candidate numbers, hardware, shapes, dtype, sample counts and costs for
LRU, SLRU, TP2 chunking, intermediate-output omission and C500 graph modes.
Its [measurement export](validation/performance-ablation.json) contains
per-run evidence. Those component experiments precede final integration;
they must not be presented as a full rerun of this revision.

A final TP2 mixed-request smoke comparison used chunk512, a 2048-token long
prompt, two 128-token short prompts, 160 output tokens, and prefix reuse off.
One window per mode measured 161.71 token/s eager and 161.02 token/s with
Decode graphs; all outputs matched. This does not demonstrate a throughput
improvement. KV-poison checks are correctness instrumentation and are not
used as performance measurements.

The earlier C500 ablation showed clearer Decode ITL benefits and only a small
increment from fixed-size Prefill graphs; see [chunk-graphs.md](chunk-graphs.md).
Chunking improves opportunities for other requests to progress, while it can
increase long-request TTFT and reduce throughput. No universal speedup is
claimed.

## Limits

PP graphs, TP2/PP2, remote-KV integration, MoE, quantization, multimodal models,
Ascend and Moore devices were not tested. This does not extend those support
claims. PP graph/chunk combinations are rejected; fixed-size Prefill graphs
are restricted to MetaX TP1. Full graph workspace memory overhead and sustained
service throughput were not measured. Near-tied FP16 tokens can differ across
prefix execution shapes, as documented in the earlier chunking checks.

Builds reused existing objects where possible. A clean build and the upstream
hardware CI matrix remain external validation steps. The native runners test
the real scheduler/model/cache path directly; they do not replace an HTTP
load test. The direct-native examples bypass the scheduler and deliberately
reject the chunking option.
