# Inference memory management

Shared static/paged KV factories allocate exact-sized storage, avoiding reusable-pool
size-class rounding. Default cache reset drains the owner stream, releases old KV,
and trims idle blocks before replacement; live graph references retain their backing.
Custom cache/reset implementations inherit only the shared paths they call.
CPU and zero-byte tensors use the existing ordinary allocation fallback.

## Execution conditions and fallback

The decoder selects optimizations by component capabilities and weight layout,
without a model-name allowlist. Existing device kernels and model constraints still apply.

| Optimization | Required conditions | Fallback |
| --- | --- | --- |
| Prefill chunking | NVIDIA BF16, unquantized weights/KV, PP1, paged attention, no EP; request history only in KV and both attention/FFN declare segmentable execution | Original decoder execution |
| Embedding sharding | NVIDIA BF16 unquantized weights, PP1, TP>1, hidden dimension divisible by TP | Replicated embedding |
| Head sharding | NVIDIA BF16 unquantized weights, PP1, TP>1, vocabulary divisible by TP, BF16 head output | Replicated head |

Unknown components default to no chunking. Derived components that change state/token
semantics must override inherited capabilities. Recurrent state and undeclared MoE
paths do not automatically opt in. Graph recording, visual/target-hidden inputs,
NSA ownership, or missing packed-request metadata use the original forward path.
Sharding remains independent of attention backend and KV quantization.

Chunking preserves absolute positions, prior KV, request mappings and public tensor
shapes. It bounds decoder intermediates; whole-input embeddings and final hidden
storage remain. BF16 execution-shape changes can affect rounding and near-tied argmax.
Sharding preserves checkpoint keys and head scaling, using TP communication to
restore complete public outputs. Sampling warmup is restricted to NVIDIA TP.

These are eligibility fallbacks, not exception recovery: an OOM, unavailable kernel,
or invalid configuration remains an error. Disabling optimization may require more memory.
Shared KV allocation/reset has no model, dtype or NVIDIA gate; other accelerators
reuse the existing Runtime allocator and synchronization APIs but are not newly qualified.

## Configuration

Optional fields in the model's `config.json`:

```json
{
  "infinilm_prefill_chunk_size": 4096,
  "infinilm_tp_vocab_sharding": true,
  "infinilm_tp_embedding_sharding": true
}
```

Chunk size `0` disables chunking; valid values are integers in `[0, INT32_MAX]`.
The head flag must be boolean and also controls embedding when the embedding flag
is absent. An explicit boolean embedding flag independently overrides that default.

## Minimal regression commands

```sh
xmake build test_kv_cache_storage
xmake build test_post_load_traversal
xmake run test_kv_cache_storage --cpu
xmake run test_kv_cache_storage
xmake run test_post_load_traversal --memory-config-only
xmake run test_post_load_traversal --cpu-cache-only
INFINILM_RUN_CUDA_TESTS=1 INFINILM_TEST_TP_SIZE=1 \
  python -m unittest discover -s test/runtime -p test_inference_memory.py -v
INFINILM_RUN_CUDA_TESTS=1 INFINILM_TEST_TP_SIZE=2 \
  python -m unittest discover -s test/runtime -p test_inference_memory.py -v
```

Expose one/two idle CUDA devices for the respective runs. Default CUDA skips do not
qualify hardware. Tiny deterministic weights require neither checkpoints nor Torch.
Tests cover actual shard shapes, independent flags, FP32/indivisible-shape fallback,
Q/K normalization, biased QKV, scaled heads, flash attention, chunk boundaries,
existing KV, mixed requests, B1/B8/B16 prefill/decode, and exact KV graph lifetime.
Host policy assertions enumerate all device types without initializing their SDKs;
they verify gating rather than execution on those accelerators.

## Measured effect and limits

Qwen3-32B BF16, TP2/PP1, two A100 40GB, B1 x 10240 input, 64 output tokens,
10496 usable KV slots/card, eager; one warmup and three measured runs per process:

| Enabled | Higher-card NVML v2 peak GiB | Median generation s |
| --- | ---: | ---: |
| Neither chunking nor embedding/head sharding | 35.85 | 4.089 |
| Chunking only | 35.29 | 4.114 |
| Sharding only | 34.41 | 4.057 |
| Both | 33.85 | 4.076 |

All variants retain exact KV allocation, so this is not a main-to-PR comparison.
Both save 2.00 GiB/card (5.58%); generated IDs match. Three timing samples do not
establish speedup. A representative 32.5 MiB KV allocation uses 32.5 MiB backing
instead of 64 MiB; that ratio is not a whole-model reduction. NVML was sampled at
20 ms and includes pool/context/communication memory; brief peaks may be missed.

Hardware execution was qualified on A100 TP1/TP2, including long-document paired
logits/IDs and B16 graph decode. Other accelerators, TP4, PP2, and full benchmark/
HTTP suites are unqualified. Prior EAGER-loading NCCL initialization failed; successful
GPU runs used LAZY loading. An existing CPU Cast overlap assertion also fails on
unchanged baselines. INT8 KV and graph-workspace sharing remain separate work.
