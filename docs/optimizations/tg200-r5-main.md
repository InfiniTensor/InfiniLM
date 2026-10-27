# R5 migration to the current execution stack

The reference R5 source is preserved in
[`superme1one/InfiniLM:perf/tg200-r5-reference-20261026`](https://github.com/superme1one/InfiniLM/tree/perf/tg200-r5-reference-20261026)
at `f78b5447d498e1b4b8fc2487a8434b6cb6cdc8ab`. Its measured runtime uses the
legacy InfiniCore API. This change is based on upstream
`1b005e15ad3a0f965935a645d46f3b69abe12314` and uses the worker-owned InfiniLM
Runtime and the public InfiniOps interface.

## Component mapping

| R5 component | Current-main implementation or migration |
| --- | --- |
| Host sequence-length reuse | Already represented by validated static attention metadata. |
| Residual Add + RMSNorm | Already represented by `RMSNorm::forward_inplace` and `FusedAddRmsNorm`. |
| Paired Q/K RoPE | The profile combines the static attention layer's two calls through the existing shared RoPE pair overload. The paired 64-thread head layout is migrated in InfiniOps. The old preload symbols are not used. |
| Static vocabulary projection | Added as an opt-in for the R5-qualified BF16 shapes. Existing packed-prefill projection remains in place. |
| Batch EOS | Added as an opt-in for low-level static text generation. |
| Contiguous Add and SwiGLU indexing | Migrated to the public InfiniOps `Add` and `SiluAndMul` providers. |
| Softmax launch sizing | Migrated to the public `Softmax` provider used by current main. The deprecated causal operator is not reintroduced into the model. The new provider retains current-main FP32 accumulation. |

The historical R5 results cannot be used as the performance result of this
port. The operator interface and floating-point implementations changed
upstream. Qualification must compare the port against current main and also
record the preserved R5 as a separate reference.

## Opt-in static profile

Pass `static_r5_profile=True` when constructing `InferEngine` to enable
the static vocabulary projection and paired Q/K RoPE. The default is `False`.
The pair call is restricted to BF16 Llama with 28 Q/K heads of dimension 128
on Iluvatar with one tensor/pipeline rank and no quantization configuration;
paged attention and graph recording retain their existing calls.

The projection shortcut also excludes requests that ask for all positions,
paged input, graph recording, other devices and parallel/quantized execution.

The qualified configuration is a Llama model with BF16 weights, hidden size
3584 and vocabulary size 73448 on Iluvatar, with no LM-head bias or output
dtype override. The original R5 shape restrictions are retained:

| Batch | Prompt length | Projection per request |
| --- | --- | --- |
| 4 or 16 | 2–64 | Final hidden-state row only |
| 4 | 512 | A 64-row zero-padded tile, retaining the final row at position 63 |
| 16 | 2048 | A 64-row zero-padded tile, retaining the final row at position 63 |
| Other shapes | Any | Existing main behavior |

Padding preserves the position within the original 64-row GEMM tile used in
R5. It avoids projecting every prompt position while retaining the validated
tile geometry. It does not establish numerical equivalence for an arbitrary
GEMM provider. The full hidden-state output remains available.

## Batch completion

Use `GenerationConfig(stop_batch_on_eos=True)` together with `stop_on_eos=True`.
A row remains marked complete after its first EOS, and generation stops when
every row has completed. Rows are not removed or reordered. Returned steps
retain the existing rectangular layout; callers should still trim each row at
its first EOS. The check is disabled for paged, Mamba-cache and multimodal paths.
Use `stop_on_eos=False` for fixed-workload performance measurements.

## TianShu build notes

Build InfiniRT, InfiniOps and InfiniCCL from the pinned Core submodules into one
private prefix with the COREX-compatible C++ ABI. For the combined profile,
replace the Ops build with [InfiniOps PR #1010](https://github.com/InfiniTensor/InfiniOps/pull/1010)
at `6489ac72cd99c487c11c4f0d582c2c12a4f3150a`. The InfiniLM submodule pins are
unchanged; this companion dependency must be integrated upstream before the
stock dependency build can provide the combined result. Do not combine the
legacy R5 libraries with this module. Install the public operator API needed by the LM
source, including explicit API-only entries for unsupported model families;
API-only entries do not provide executable kernels for those models.

The tested operator selection is recorded in
[`tg200-r5-sdk-ops.json`](tg200-r5-sdk-ops.json). Configure the companion Ops
build with `WITH_CPU=ON`, `WITH_ILUVATAR=ON`, `WITH_TORCH=ON`,
`AUTO_DETECT_DEVICES=OFF`, `AUTO_DETECT_BACKENDS=OFF`,
`INFINI_OPS_OPS=/absolute/path/to/tg200-r5-sdk-ops.json`, and
`INFINI_OPS_TORCH_OPS=argmax,tril,triu`. Point `INFINI_RT_ROOT` and
`CMAKE_INSTALL_PREFIX` at the same private prefix and select C++ ABI 0 for
the tested COREX PyTorch build. The stock dependency pins do not contain the
companion capability-query fix or these R5 kernels.

COREX 4.3 includes an older copy of pybind11. Use the pinned pybind11 3.0.4
headers before the SDK include directory, and include the active Python's
development headers. An explicit `xmake f --includedirs=/path/to/pybind11/include`
can set that priority; `CPLUS_INCLUDE_PATH` can provide the Python include path.
Both paths must match the selected build inputs. This avoids removing the new
destructor/GIL handling merely to compile against an old SDK header.

The checkpoint reader uses a private copy-on-write mapping with read-only
NumPy views. COREX 4.3 rejected a 135,790,592-byte H2D copy from an OS read-only
mapping in the qualification environment; the same bytes copied successfully
from a private mapping. The original checkpoint file is never modified.

The legacy 9G checkpoint contains Q/K/V bias tensors but omits
`attention_bias` from its configuration. Old R5 defaulted it to `true`;
current main defaults it to `false`. Set the field to `true` in a separate
configuration copy for this checkpoint, based on its actual tensor keys.
Do not change the general Llama default or silently discard those weights.

## Reproduce fixed-workload generation

Prepare a JSON matrix of token IDs and use exactly the same file, cache size,
weights and generation length for both runs. The script disables EOS early
termination and graph compilation, separates warmups from measurements,
checks repeated greedy outputs and records every output token.

```sh
python test/bench/benchmark_static_r5.py --model MODEL --input-ids ids.json \
  --device iluvatar:0 --new-tokens 8 --cache-capacity 544 --output baseline.json
python test/bench/benchmark_static_r5.py --model MODEL --input-ids ids.json \
  --device iluvatar:0 --new-tokens 8 --cache-capacity 544 \
  --static-r5-profile --output candidate.json
```

These two commands isolate the model-layer option with one SDK. To assess the
combined R5 port, repeat with the qualified InfiniOps build and record both
source revisions. Compare token matrices before interpreting timings. A match
on these workloads is not a claim of bitwise equivalence for all models or a
full task-quality evaluation.

## Results on 2026-10-27

All runs used one TG-V200, COREX 4.3, BF16 weights, TP/PP 1, greedy sampling,
disabled graphs and disabled EOS early stopping. Each configuration used two
warmups followed by three measured requests. Times include `generate` and
copying all generated token IDs to the host; the table reports the median.
The same stored token IDs and cache capacities were used for all three roles.

| Model / batch / input / output / cache | Modern baseline (ms) | Migrated R5 (ms) | Latency reduction | Preserved legacy R5 (ms) |
| --- | ---: | ---: | ---: | ---: |
| 9G7B_MHA / 4 / 512 / 8 / 544 | 839.090 | 761.416 | 9.26% | 612.914 |
| 9G7B_MHA / 16 / 2048 / 16 / 2080 | 12558.310 | 11338.654 | 9.71% | 8108.814 |
| 9g_8b_thinking_llama / 1 / 128 / 8 / 160 | 369.676 | 357.037 | 3.42% | 330.824 |

Every measured and warmup request produced the same token IDs across all
three roles. The second model uses the model-layer fallback, so its result
must not be attributed to the vocabulary projection or paired MHA RoPE.
The migrated combination improves this modern baseline but remains slower
than the preserved legacy R5. This port does not restore the old runtime's
performance; the legacy branch remains available and is not replaced.

The baseline is upstream `1b005e1` with its pinned Ops `1ca4777`, the private
checkpoint mapping fix and the Iluvatar capability-query fix. Those fixes
are needed to execute this checkpoint with the modern stack on COREX.
The candidate uses the same InfiniLM base plus this PR and companion Ops
`6489ac7` based on current Ops `master`. This measures the combined migration,
not an ablation that attributes every millisecond to an individual kernel.
Individual samples, revisions and input/output hashes are recorded in
[`tg200-r5-results.json`](tg200-r5-results.json); original token inputs and
model weights are not distributed.

Additional validation:

- 149 public-SDK operator cases passed and matched baseline output byte hashes.
- 14 static-shape cases matched generated tokens: batches 4/16, input lengths
  1/2/7/63/64/65/128, eight generated tokens, and distinct input rows.
- Batch EOS integration passed at batches 4 and 16; 14 host tests cover
  staggered completion, multiple EOS IDs, disabled/default and excluded paths.
- The projection-policy C++ boundary test and checkpoint mapping host test passed.
- `examples/test_infer.py` passed on the real checkpoint.
- `examples/bench.py` passed after switching its stale external InfiniCore
  import to the bundled runtime. The standalone fixed-workload CLI also passed.
- `test/bench/test_benchmark.py` completed the first ten MMLU abstract-algebra
  test rows from an existing local dataset, converted to an offline Arrow
  cache. Baseline and candidate both scored 6/10 with the same predicted
  choices. This is a ten-question regression, not a full MMLU result.
- The HTTP server returned a complete SSE stream; `scripts/test_perf.py`
  completed four requests at concurrency two with eight output tokens each.
  This is a service smoke test, not a capacity or tail-latency study.

The high-level single-request, MMLU and HTTP tests exercise default model
behavior with the companion SDK. The opt-in model profile is exercised by
the fixed-workload CLI and the static-shape comparisons above.

The MMLU sanity entry must load the native runtime before `datasets` imports
Torch on this COREX stack. The import-order fix avoids a native-module import
stall and is kept separate from the inference changes.

One existing checkpoint test still fails on both baseline and candidate:
`test_checkpoint_cast_rejects_bad_contract_before_writing`, where the CPU
Cast provider accepts overlapping input/output storage with different dtypes.
The other 11 checkpoint tests passed. This unchanged baseline defect is not
reported as a successful full checkpoint suite. NVIDIA and other vendor
hardware were not available for local execution; platform CI requires
maintainer approval for this fork's contribution.
