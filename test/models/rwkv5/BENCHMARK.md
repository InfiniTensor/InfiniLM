# RWKV5 NVIDIA Benchmark

## Environment

- GPU: NVIDIA GeForce RTX 4090 D, 24 GiB
- Model: RWKV-5-World-0.1B-v1-20230803-ctx4096
- Precision: BF16
- Cache mode: paged, 128 blocks, block size 16
- Sampling: greedy (`top_k=1`), EOS ignored to keep output lengths identical
- Warmup: 3 runs
- Measured runs: 10 per comparison case

The prompt token IDs are deterministic and identical between the two modes. The
comparison is an architectural ablation: the normal RWKV5 path uses only one
recurrent state row per request, while the forced path exercises the generic
attention-KV scheduler path. RWKV5 itself does not consume attention KV blocks.
This benchmark does not claim to change the generic KV admission or
over-reservation policy used by Transformer models.

## Metric Definitions

- TTFT is batch max TTFT: time from submitting the batch until every request has
  produced its first token.
- Decode throughput counts only tokens generated after every request has reached
  TTFT. Tokens produced by earlier requests while another request is waiting are
  not counted twice.
- End-to-end throughput is total requested output tokens divided by total elapsed
  generation time.
- P50 is the median. P90 uses the nearest-rank definition.

## Comparison

Batch size is 4 and each prompt contains 512 tokens.

| Output tokens | Mode | TTFT P50 | TTFT P90 | E2E throughput P50 |
|---:|---|---:|---:|---:|
| 32 | Forced attention-KV path | 337.22 ms | 362.65 ms | 254.95 tok/s |
| 32 | Recurrent-only state path | 120.32 ms | 120.98 ms | 362.90 tok/s |
| 128 | Forced attention-KV path | 1017.99 ms | 1065.04 ms | 308.07 tok/s |
| 128 | Recurrent-only state path | 120.08 ms | 120.27 ms | 474.20 tok/s |

Results:

- Output 32: TTFT P50 decreased by 64.32% (2.80x), while end-to-end
  throughput increased by 42.34%.
- Output 128: TTFT P50 decreased by 88.20% (8.48x), while end-to-end
  throughput increased by 53.93%.
- The forced attention-KV path generated 97 or 385 tokens before all requests reached TTFT. This
  means the first three requests completed before the fourth request started.
  The recurrent-only path generated exactly four tokens at the same boundary, one
  per request, confirming that RWKV5 does not request attention KV blocks.
- Peak GPU memory was 1203 MiB in both modes. The architectural path avoids
  attention-KV allocation for RWKV5; it does not change the generic KV
  reservation policy or the RWKV state pool allocation.

## Reproduction

Recurrent-only state path:

```bash
RWKV5_MODEL_PATH=/models/RWKV-5-World-0.1B-InfiniLM \
python test/models/rwkv5/benchmark.py \
  --cache-type paged --num-blocks 128 \
  --batch-sizes 4 --input-lens 512 --output-lens 32,128 \
  --warmup 3 --runs 10
```

Forced attention-KV ablation:

```bash
RWKV5_MODEL_PATH=/models/RWKV-5-World-0.1B-InfiniLM \
python test/models/rwkv5/benchmark.py \
  --cache-type paged --num-blocks 128 \
  --batch-sizes 4 --input-lens 512 --output-lens 32,128 \
  --warmup 3 --runs 10 --force-attention-kv
```

Raw results:

- `results/rwkv5-p50-optimized-v2.jsonl`
- `results/rwkv5-p50-legacy-v2.jsonl`
- `results/rwkv5-full-optimized-p50.jsonl`

The full optimized matrix covers batch sizes 1, 2, and 4; input lengths 32,
128, and 512; and output lengths 32 and 128, with five measured runs per case.
