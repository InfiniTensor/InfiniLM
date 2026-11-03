# Ascend fresh-prefill graph regression

`fresh_prefill_graph.cc` exports `check_fresh_prefill_compiler()` for a native
TP4 regression. Build it as a shared library against the current InfiniCore and
InfiniLM extension, then call the exported function through ctypes. It requires
four visible Ascend devices and the same Ascend/InfiniCore library environment
as InfiniLM.

Enable `INFINILM_ASCEND_FRESH_PREFILL_GRAPH=1`,
`INFINICORE_ASCEND_ACLNN_CACHE=1`, `INFINICORE_ASCEND_APPLY_ROPE=1`,
`INFINICORE_ASCEND_APPLY_ROPE=1`, `INFINICORE_ASCEND_CACHE_SLOT_VALIDATION=1`,
`INFINICORE_GRAPH_SYNC_WARMUP=1`, and `HCCL_OP_EXPANSION_MODE=AIV`.

The four-layer fixture exercises strided K/V attention, last-token selection,
changing tokens/positions/cache slots, and full KV-cache comparison against
operator-by-operator execution at 1024, 2047, and 2048 tokens. Between requests it
explicitly overwrites every cached device input and clears ephemeral host
bindings to emulate decode reusing the graph allocator arena. This detects
missing refreshes of shape-constant offsets/lengths and reliance on stale host
metadata. Unsupported input shapes, prefix hits, duplicate slots and speculative
outputs must fall back. This is a correctness test, not a performance estimate.

Real model logits must additionally be compared with identical teacher-forced
inputs before accepting a performance result. This fixture does not replace
real GEMM, normalization and FFN coverage.


## Ascend native decode-cache regression

`paged_decode_graph.cc` exports `check_validated_decode_compiler()` for TP4 with
`INFINICORE_ASCEND_ACLNN_CACHE=1` and
`INFINICORE_ASCEND_DECODE_CACHE=1`. The latter is an explicit opt-in for
native ACLNN decode cache writes at exact batch sizes 4 through 16.
The fixture covers batches 1, 4, 8 and 16, strided K/V inputs, changing token
IDs and unique cache slots across five replays, and full CPU cache/output
comparisons. Duplicate, negative, out-of-capacity, missing, wrong-dtype and
wrong-length slots must use the existing safe fallback before graph replay.
Batch 1 retains its original path. The graph owns each replay's host slot
metadata; it must never depend on an expired Python input buffer.

## Ascend NZ prefill communication overlap

The Core fixture `test/infinicore/ops/linear_nz_allreduce_graph.cc` exercises
`INFINICORE_ASCEND_PIPELINED_NZ_REDUCE=1` with four distinct weights per rank,
five changing inputs, bias and no-bias cases, CPU references and the original
GEMM/all-reduce implementation. Cases cover the 1024-row boundary, 2047/2048
rows, K=2048/7392 and N=8192. The native graph uses Ex events to connect the
main compute stream to one shared communication stream per rank and joins it
before the next dependent operation. Eager and unsupported layouts retain
the original implementation. This flag defaults off.

After either change, validate the real four-layer model and full model with
identical teacher-forced inputs before timing the service. Full-model coverage
includes a 16-request sequence crossing a 256-token cache-page boundary.
Process-exit HCCL destruction errors observed in the diagnostic Python harness
are recorded separately; successful logits checks do not establish clean
teardown.


## Ascend sampled-token transfer

`ascend_token_output.cc` exports `check_async_token_output()`. Compile as a
shared library against InfiniCore, initialize the Core runtime, and invoke the
export with four visible Ascend devices. It checks batches 1/4/8/16 and the
17-token fallback against 32 changing BF16 score tensors (vocab 73448), including
first-index ties and CPU results retained across later calls. The test must
restore the NPU context before a new H2D: allocating a regular CPU tensor
changes the current Core runtime to CPU. Interleaved micro timings include
sampling and host transfer and do not replace service measurements.

`INFINILM_ASCEND_ASYNC_TOKEN_D2H=1` enables the worker path for Ascend greedy
sampling with PP=1, contiguous int64 token IDs and at most 16 outputs. A
per-worker-thread pinned staging buffer receives an InfiniRT asynchronous D2H
on the current computation stream, after the sampler. One stream synchronization
precedes CPU consumption. The returned CPU tensor owns separate storage, so
later tokens cannot overwrite earlier results. Unsupported cases retain the
original synchronization/copy path. The flag defaults off; no model arithmetic
or sampling policy changes. Validate real four-layer and full-model logits,
changing generated IDs, and natural-language service outputs before accepting
full-length performance measurements.
