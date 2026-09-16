# Cache, chunking and graph performance evidence

These results explain the individual mechanisms integrated by this PR. They
are archived component experiments, **not a complete performance rerun of the
final integrated revision**. Final integration correctness is reported in
[cache-chunk-validation.md](cache-chunk-validation.md). Do not multiply the
gains from different experiments or interpret them as a universal speedup.

[Sanitized measurements](validation/performance-ablation.json) include
per-run values, original artifact names and SHA256 hashes. The values below
were recalculated from request accounting, output timestamps and recorded
execution steps. The export omits machine paths, credentials and prompts.
Historical source commits identify development snapshots; they are not the
published squash commit. Worktree/binary hashes in the original experiments
were used where a commit alone did not describe the tested build.

[Results screenshot](validation/test-results.png): browser capture of the
audited saved-results report, not a new GPU run or upstream CI result.

## Definitions and common conditions

- TTFT: request submission to its first generated token. ITL: interval between
  consecutive generated tokens. An active request's maximum ITL measures its
  worst observed pause, not its typical token latency.
- Cached tokens are actual admitted reused tokens; Prefill tokens are prompt
  tokens that still require computation. Warmup requests are excluded below.
- Percentage change is `100 * (new / old - 1)`. Negative latency or work
  changes are reductions. Each table names its baseline and aggregation.
- All model experiments use dense models, greedy decoding and paged KV with
  256-token pages. Fixed output budgets ignore EOS for timing; normal-EOS
  and lifecycle checks are separate. No sustained HTTP load, confidence
  intervals or statistical significance claim is provided.
- NVIDIA experiments use RTX A6000 48 GB GPUs, CUDA 12.4 and driver 580.105.08;
  Qwen2.5 model/KV tensors are FP16. GPU(s) were checked for availability,
  but the host was shared. TP2 uses NCCL on the two local A6000s.

## LRU versus the former set-based reclamation

One A6000, Qwen2.5-1.5B (model revision
`8faed761d45a263340a0528343f099c05c9a4323`), TP1, eager paged attention,
64 cache blocks, concurrency 1. Each repeat has 128 requests, of which the
first 16 warm the cache; each prompt has 1,024 prefix tokens plus a unique
32-token tail, and produces 32 tokens. Two independent engine repeats use
the same hotspot/cold trace. The baseline is `270feb3e`; the LRU component
is `49410568`, both using Core `1ab85ef1`.

| Metric | Set-based baseline | LRU | Interpretation |
|---|---:|---:|---|
| Cached tokens per repeat | 61,440 | 86,016 | +40.00% reused work |
| Prefill tokens per repeat | 56,832 | 32,256 | **43.24% less prompt computation** |
| Mean TTFT, repeats 0 / 1 (ms) | 38.37 / 38.46 | 27.59 / 27.54 | Median paired reduction **28.24%** |
| Output token/s, repeats 0 / 1 | 120.94 / 124.60 | 121.98 / 124.99 | +0.87% / +0.32%; no stable throughput gain |

The policy retains recently reused prefixes instead of depending on set
iteration order. It is not universally better: Qwen2.5-7B with a 128-page
cyclic working set and only 64 blocks had 33,792 -> 0 cached tokens and
34.96 -> 33.16 output token/s (**5.14% lower**, one run). LRU thrashed on
that trace while the old policy happened to keep a subset.

Some historical FP16 cache-on/off outputs differ with execution shape.
Matching-retention controls and logit replays were investigated separately;
these latency results do not establish universal token-level equivalence.
The final integrated correctness checks also retain this limitation.

CPU-only method-call microbenchmark: Python 3.11.15, 65,536 blocks, 90% pinned,
five fresh pressure pools. Reclaiming one block took a median **2,058.758 us
-> 18.217 us** (113x lower call time). Pool construction and hashing were
excluded. Retained Python metadata increased by **5,767,328 bytes** at that
pool size. This is not GPU time or an end-to-end model speedup.

## Optional SLRU versus LRU under a one-use scan

One A6000, the same 1.5B FP16 model, TP1, eager, concurrency 1, eight blocks,
protected ratio **0.5** (the API default is 0.8). Four warmup requests establish
hot prefixes, followed by ten measured 544-token requests, eight output
tokens each. One run per configuration; the trace includes six one-use
prefixes. All 14 output arrays and normal-EOS checks match across LRU8,
SLRU8, LRU32 and cache-off controls.

| Metric | LRU, 8 blocks | SLRU, 8 blocks |
|---|---:|---:|
| Cached tokens | 1,024 | 2,048 |
| Prefill tokens | 4,416 | 3,392 (**23.19% less**) |
| Observed mean TTFT (ms) | 28.26 | 23.91 |

SLRU8 matches the retained-token count of LRU32 in this trace. It preserves
established hot prefixes during a scan; it does not increase physical KV
capacity. No stable throughput claim follows from this one short run.
Scheduler-only no-reuse and overcapacity-cycle traces have zero hits with
both policies; the tested hotspot-shift trace has no total-work improvement.

## TP2 chunked Prefill versus unchunked eager

Two A6000s, Qwen2.5-1.5B FP16, TP2/PP1, eager paged attention, 128 cache blocks,
prefix reuse off, max batch size 2. A 128-token prompt generates 128 tokens;
after its eighth token an 8,192-token prompt generating 16 tokens arrives,
then a second 128-token prompt generating 16 tokens arrives 20 ms later.
A separate warmup precedes measurement. Values are medians of three short
runs per setting; all 160 output IDs match across the nine runs. Component
snapshot: `18beeeec` plus the recorded working-tree implementation.

| Metric | Unchunked | Chunk512 | Change | Chunk1024 |
|---|---:|---:|---:|---:|
| Active request maximum ITL (ms) | 1,022.94 | 142.74 | **-86.05%** | 239.21 |
| Late short request TTFT (ms) | 1,001.54 | 68.16 | **-93.19%** | 139.44 |
| Active request p95 ITL (ms) | 24.88 | 92.68 | +272.47% | 63.62 |
| Long request TTFT (ms) | 1,012.17 | 1,435.76 | +41.85% | 1,210.33 |
| Finite-window output token/s | 75.27 | 62.83 | -16.53% | 68.06 |

Chunk512 allowed 15 active Decode steps during the long Prefill, and the late
short request produced its first token before the long Prefill finished.
The gain is bounded worst-case blocking and request progress. More frequent
short pauses replace one long pause, so p95 ITL can worsen while maximum ITL
improves. Chunk1024 reduces maximum pause by 76.61% with a 9.58% output-rate
cost. Chunk size is an application latency/throughput choice.

Skipping unused intermediate LM-head/sampling/output work was compared
separately on this chunk512 workload, with three alternating runs per mode.
Median cumulative long-Prefill dispatch time changed **1,202.99 -> 1,175.69 ms
(-2.27%)**, while output rate changed **64.19 -> 62.87 token/s (-2.05%)**.
This does not establish a serving-throughput gain. Every Transformer layer,
including Attention, MLP and required collectives, remains fully executed.

## C500 graphs: same-binary five-mode ablation

MetaX C500 **50% compute / 32,000 MiB slice**, CPU quota six cores; MACA
3.5.3.20, driver 3.8.30, torch 2.8.0+metax3.5.3.9, Flash Attention
2.6.3+metax3.5.3.9torch2.8. Qwen3-0.6B/4B BF16, TP1/PP1, vendor Flash
Attention, 32 KV blocks, prefix reuse off. All five modes use the same LM
binary built from the pre-integration graph prototype based on `ac89c5c3`.
This isolates execution mode without mixing native/vendor backends.

Single requests: 2,048 input / 16 output tokens, three measurements after
warmup; TTFT is their mean, ITL is the mean of per-request median intervals.
Mixed windows: a 128-input / 48-output request receives a 2,048-input /
16-output competitor after its eighth output token; two windows per mode.
Window rate is 56 remaining output tokens divided by time from long-request
admission until both finish. It is not sustained serving throughput.

| Model | Mode | Single TTFT ms | Single ITL ms | Mixed short max ITL ms | Mixed window token/s |
|---|---|---:|---:|---:|---:|
| 0.6B | Unchunked eager | 77.79 | 6.86 | 85.60 | 154.26 |
| 0.6B | Unchunked + Decode graph | 77.34 | 4.26 | 82.53 | 226.68 |
| 0.6B | Chunk512 eager | 97.22 | 6.82 | 34.58 | 145.96 |
| 0.6B | Chunk512 + Decode graph | 96.83 | 4.26 | 31.56 | 209.73 |
| 0.6B | Chunk512 + Prefill/Decode graphs | 91.01 | 4.37 | 30.21 | 212.86 |
| 4B | Unchunked eager | 211.94 | 12.99 | 227.82 | 72.79 |
| 4B | Unchunked + Decode graph | 211.38 | 9.39 | 222.93 | 93.35 |
| 4B | Chunk512 eager | 248.52 | 12.96 | 82.73 | 69.80 |
| 4B | Chunk512 + Decode graph | 247.97 | 9.37 | 77.40 | 88.06 |
| 4B | Chunk512 + Prefill/Decode graphs | 241.37 | 9.37 | 75.52 | 88.93 |

At chunk512, Decode graphs reduce 4B single-request ITL by **27.64%** and
increase the mixed-window output rate by **26.16%** versus chunk eager.
Adding experimental Prefill graphs then reduces TTFT by **2.66%** for 4B
and **6.01%** for 0.6B. That small increment is not a significance claim.
The 4B full combination versus unchunked eager reduces the competing short
request's maximum pause by **66.85%**, with **22.17%** higher window output
rate; single-request TTFT is still higher (211.94 -> 241.37 ms).

All 30 single-request outputs match the HF eager reference; all 20 mixed
windows match their long-request reference and short-request eager control.
Actual device-graph launches were counted, not inferred from an enable flag.
Existing Decode graph infrastructure is reused; this PR's contribution is
its safe combination with chunking and experimental fixed-size Prefill graphs.

Costs: 4B initialization, including weight loading and graph preparation,
was 5.08 s unchunked eager, 6.30 s chunk + Decode graph and 7.05 s with both
graphs. This is not isolated graph capture time. Extra retained graph memory
was not measured. The shared slice, small samples and same-family models
limit generalization.

## Final integrated performance check and reproduction

The final A6000 TP2 check used a shorter 2,048-token long prompt, two 128-token
short prompts, chunk512, prefix cache off and 160 total outputs. One window
per mode measured **161.71 token/s eager vs 161.02 token/s Decode graph**
(-0.43%); outputs matched. No throughput improvement is established by this
pair. Do not compare its rates directly with the 8,192-token component test.
PP2 eager has correctness/lifecycle coverage, not a performance comparison.

The committed [prefix benchmark instructions](../test/llm/README.md),
[SLRU checks](../test/llm/README.slru.md),
[chunk benchmark](chunked-prefill.md) and [graph configuration](chunk-graphs.md)
describe runnable checks and prerequisites. Match model, dtype, pool,
prefix setting, prompt lengths, warmup, output budget and repetition count
before comparing runs. The JSON export allows independent aggregation of
the archived measurements; it is not a portable archive of every historical
build or experiment driver. No GPU experiments were rerun for this report.

Remaining coverage limits: sustained service load, full graph memory cost,
PP graphs, TP2 Prefill graphs, four-GPU TP2/PP2, remote-KV/chunk combinations,
MoE, quantization, multimodal, and unconnected Ascend/Moore hardware.
