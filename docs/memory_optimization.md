# Inference memory management

Shared static and paged KV storage use exact-sized allocations instead of rounded reusable-pool
size classes. Cache capacity changes drain the owner stream and trim idle,
unpinned blocks before replacement. Live graph allocations retain their backing.
The existing parameter allocation APIs remain available.

These runtime/cache changes have no Qwen3, BF16, or TP2 enablement filter. They
apply to model paths that call the shared static/paged KV factories and, for
replacement reclamation, the default cache reset path. Custom state/cache
allocators or overridden reset methods only inherit the parts they actually
call. Existing backend/dtype/topology support still applies; this change does
not add a missing device kernel or make an unsupported model configuration run.
CPU and zero-byte allocations retain the ordinary allocation fallback.

The model-execution optimizations below are separate. Their implementation uses
shared layers/templates, but their current default enablement is explicitly
restricted by `supports_qwen3_memory_optimization`. That is a code restriction,
not a claim that other configurations merely lack benchmark results.

For ordinary Qwen3 on NVIDIA, BF16 weights and KV, PP1, TP1/2, and `paged-attn`,
prefill runs through the decoder in blocks of at most 4096 tokens. Each block
keeps the original absolute positions, request page mapping, and preceding KV.
Sampling and request completion occur after the complete prefill. Full returned
hidden states and logits keep their existing shapes.

TP2 also shards embedding weights along the hidden dimension and the output
head along the vocabulary dimension. Checkpoint keys remain unchanged. The
existing AllReduce contract restores the full public embedding/logits tensors.
The FP32 output-head option retains the replicated head. A vocabulary that
cannot be divided across TP ranks retains a replicated head; an indivisible
hidden dimension independently retains a replicated embedding.

The following advanced fields in the model's `config.json` control these paths:

```json
{
  "infinilm_prefill_chunk_size": 4096,
  "infinilm_tp_vocab_sharding": true
}
```

Set the chunk size to `0` to keep the original prefill execution shape. Set
vocabulary sharding to `false` to retain replicated embedding/head weights.
Disabling either optimization can increase memory requirements. Chunk size must
be a nonnegative integer no larger than `INT32_MAX`; sharding must be a boolean.

For model-internal chunking and embedding/head sharding, other models, devices,
FP16/FP32 model weights, quantized paths, PP>1, and TP>2 retain their previous
execution paths. They can still use the shared exact KV allocation and default
cache replacement improvements described above. Model-internal chunking also falls back
for graph recording, visual inputs, NSA ownership, and target hidden states.
This qualification scope does not certify every combination supported by the
underlying framework.

Chunking changes GEMM/attention shapes. BF16 rounding can change near-tied
argmax choices; arbitrary-input bitwise equality is not promised. Use the
opt-out settings for execution-shape comparisons.

## Regression commands

Use the formatter versions pinned by `.github/workflows/ci_test.yml`:
`clang-format==21.1.8` and `ruff==0.15.20`. Host checks do not require a native
extension or CUDA device:

```sh
python scripts/format.py --ref origin/main --check
python -m unittest discover -s test/scripts -p test_base_config_device_detection.py -v
```

The CUDA tests below are opt-in. A skipped default run does not qualify a GPU
configuration; set the opt-in flag and expose the requested number of devices.

Build the native targets using the pinned InfiniCore prefix:

```sh
xmake build test_kv_cache_storage
xmake run test_kv_cache_storage
xmake build test_post_load_traversal
xmake run test_post_load_traversal --memory-config-only
INFINILM_RUN_CUDA_TESTS=1 INFINILM_TEST_TP_SIZE=2 \
  python -m unittest discover -s test/runtime -p test_qwen3_prefill_memory.py
```

Repeat the Python test with TP1 and a single visible CUDA device. The tests use
deterministic tiny weights without downloading a model or importing Torch.
They cover block boundaries, one-token tails, packed requests, existing KV,
mixed prefill/decode, opt-outs, the FP32 output-head fallback, and indivisible
vocabulary/hidden dimensions (including the final vocabulary token).

`test/bench/check_qwen3_memory_quality.py` accepts external real-document cases
and a local checkpoint. Run it in separate processes with and without
`--control`, then optionally with `--framework vllm` for an independent reference.
It saves generated answers and first-prefill logits or reference logprobs.
`COMPLETE` means collection finished; numerical and answer comparisons must be
performed separately. Ordinary exceptions record a failure; partial files from
a terminated process must be interpreted with its exit code and logs.
The external case format is a JSON list containing `dataset`, `index`, `id`,
`prompt`, `token_ids`, `token_count`, `answers`, and `max_tokens`.
No dataset, checkpoint, server address, or hardware identifier is bundled.

Compare completed native captures on CPU, with NumPy and no model/CUDA:

```sh
python test/bench/compare_qwen3_memory_quality.py \
  --control control.json --optimized optimized.json \
  --require-identical-tokens --result comparison.json
python -m unittest discover -s test/scripts -p test_memory_quality_comparison.py -v
```

Keep each JSON next to its same-stem NPZ. The comparison rejects incomplete or
mismatched captures, missing logit vectors, invalid shapes, and nonfinite values.
Numerical acceptance requires relative L2 below 0.02 and cosine above 0.999.
Greedy token differences are reported separately; the optional strict flag also
requires identical generated IDs. This check covers the first-generation logit
vector and generated IDs, not all decode logits or cross-engine equivalence.

## Capacity and profiling

Choose the global KV pool from the total simultaneously resident token budget.
Reducing the pool reduces supported concurrent token capacity; it is not a free
memory saving at unchanged maximum concurrency and length. Account separately
for page tails, runtime/communication memory, prefill workspace, and graphs.

Compare frameworks with the same checkpoint, precision, hardware, usable KV
capacity, batch limit, and execution mode. Report startup, inference, and idle
residency separately. NVML v1 includes driver-reserved memory on the tested
driver; NVML v2 reports that component separately. A sampling maximum can miss
short-lived spikes and does not measure only live tensors.

Further graph bucket/workspace sharing, smaller KV pages, and streaming
embedding are separate follow-ups. This change does not introduce INT8 KV.

## Validation boundary

The qualification machine used two A100 40 GB devices, ordinary Qwen3-32B,
BF16 weights/KV, TP2 and PP1. Five native regression executables, the quantization
configuration guard, ten unaffected raw-loading tests, and the original three CUDA
regressions at each of TP1 and TP2 passed in the earlier qualification. The CUDA regressions use tiny random
weights; TP1 here does not mean a BF16 32B checkpoint fits on one 40 GB device.

Six fixed complete documents from public LongBench QASPER/QMSum were tested with
custom short-answer/summary prompts and identical chat tokenization: 9744, 10264,
15767, 9999, 12512 and 16250 input tokens. With optimization disabled versus
4096-token prefill plus vocabulary sharding, full first-generation logits were
bitwise identical and all six generated token sequences were identical. This is
a small real-document regression, not an official full LongBench score or a
promise of bitwise equality on arbitrary BF16 inputs. Unchunked controls at
20085 and 21932 input tokens ran out of device memory with their fixed KV budgets;
those incomplete runs are capacity observations, not precision comparisons.

One existing raw-loading test, `test_checkpoint_cast_rejects_bad_contract_before_writing`,
fails its CPU Cast overlap rejection assertion in this build and in two unchanged
baseline installations. The test is preserved and its failure is not counted as
a pass. Public generation was tested directly rather than through each example command.
The repository-wide suite and HTTP service suite were not run in full.
Other accelerators, quantization,
TP4 and PP2 were not GPU-qualified by this change.

The independent vLLM 0.11.0 run used the same six prompt token sequences and
matched the first generated token in all six cases. Full generation is not
identical across engines. On this small sample, normalized QASPER word-F1 was
17.73 with either InfiniLM configuration and 26.38 with vLLM; normalized QMSum
word-LCS-F1 was 19.12 versus 20.32. These custom three-case averages do not
establish general model quality. Maximum first-token top-20 log-probability
differences ranged from 0.249 to 1.000 and were identical before and after this
optimization. Cross-engine numerical differences remain a separate validation
question; the optimization did not introduce them in these cases.

## Measured capacity

Final-binary runs used Qwen3-32B BF16, TP2/PP1, two A100 40 GB devices, no offload
or prefix caching, one warmup and three measured generations of 64 output tokens
per request. NVML v2 whole-device memory was sampled every 20 ms; peaks below
exclude driver-reserved memory. The vLLM 0.11.0 reference reuses the completed
same-device/checkpoint comparison with equal usable BF16 KV capacity; its
16-token null block adds 2 MiB of physical KV storage. Native KV pages are 256
tokens. Fixed KV budgets must be included when comparing resident memory.

| Workload | InfiniLM peak GiB/card | vLLM peak GiB/card | InfiniLM median batch s | vLLM median batch s |
| --- | ---: | ---: | ---: | ---: |
| B1 x 40896, eager | 38.0359 | 38.4578 | 13.748 | 13.098 |
| B8 x 4096, graph | 37.8875 | 37.9598 | 8.405 | 7.063 |
| B16 x 2048, graph | 38.6883 | 38.2664 | 8.053 | 6.946 |

Each graph run verified that all 63 decode forwards used a device graph with
zero host segments and the intended simultaneous batch. Four repetitions kept
identical greedy outputs within each workload. Mapped native library hashes
matched the tested build. Timings include scheduler/generation work and are not
per-kernel comparisons. The table demonstrates comparable residency, not a
throughput advantage over vLLM. B16 graph residency remains about 0.422 GiB
higher; graph bucket/workspace sharing is a separate follow-up.

## Shared-path cross-model coverage

The deterministic CUDA fixture also constructs Llama and Qwen2 models, including
Qwen2 QKV biases, and exercises packed/mixed requests, existing KV, cache resets,
and continuation at TP1 and TP2. Both model families keep replicated embedding
and head weights even when the Qwen3 optimization flags are set. This checks
shared-cache compatibility and real enablement boundaries, not pretrained-model
quality or model-specific chunking/sharding support for those families.

The added family-specific TP1/TP2 runs passed. The combined TP1 suite passed all
four tests. A fresh combined TP2 run stalled in its original Qwen3 first case;
an EAGER-module-loading attempt exited during NCCL initialization. Those attempts
are recorded as unsuccessful, separately from the passing family-specific runs
and the earlier Qwen3 TP2 qualification. The failure occurred before the new family
branches ran; its root cause remains open.
