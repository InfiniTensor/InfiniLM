# DeepSeek-V4 Attention Optimization Log

This document records accepted stages, rejected experiments, build results,
correctness checks, and performance measurements for the DeepSeek-V4 attention
refactor.

## Measurement Rules

- Primary metrics: Prefill TTFT, Decode Avg ITL, and Decode Throughput.
- Use `run_bench.sh` for the full-model TP=8 result.
- Compare medians when three runs are available.
- A stage is accepted only when the full model still produces coherent output.
- Keep a runtime switch or a clean legacy path until the replacement is verified.

## Baseline

Date: 2026-07-11

- Branch: `dpv4-wang`
- InfiniLM commit: `0641224`
- Model: `/data/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8`
- TP: 8
- Batch size: 1
- Input tokens: 12
- Output tokens: 160
- Prefill TTFT: 2997.23 ms
- Prefill throughput: 4.0 tok/s
- Decode Avg ITL: 117.66 ms
- Decode throughput: 8.5 tok/s
- Result: coherent DeepSeek response produced successfully.

## Stage 1 - Attention State and Step Refactor

Status: accepted

Goal: move mutable cache ownership and lifecycle out of
`DeepseekV4Attention` without changing mathematical behavior or kernel dispatch.

Changes:

- Added `DeepseekV4AttentionStep` and `DeepseekV4AttentionState`.
- Moved cache initialization, reset, capacity growth, and decode append logic
  out of `DeepseekV4Attention`.
- Preserved the existing attention kernels, fallbacks, and runtime switches.

Verification:

- InfiniLM build: passed.
- Layer0-4: Prefill TTFT 516.65 ms, Decode Avg ITL 9.89 ms,
  Decode Throughput 101.07 tok/s.
- Full model: Prefill TTFT 3045.51 ms, Decode Avg ITL 116.64 ms,
  Decode Throughput 8.57 tok/s.
- Full-model output matched the baseline response and remained coherent.

Assessment: structural refactor accepted. Decode changed from 117.66 ms to
116.64 ms (-0.87%). Prefill changed from 2997.23 ms to 3045.51 ms (+1.61%),
which is within the observed full-model run-to-run noise.

## Stage 2 - Compressed Prefill GPU Path

Status: accepted

Goal: support causal/window masking for multiple query tokens in the existing
compressed attention kernel and remove the C4/HCA prefill CPU attention path.

Changes:

- Extended the InfiniCore compressed attention descriptor and kernel with
  causal masking, sliding-window masking, and a contiguous key-position base.
- Enabled the compressed GPU kernel for contiguous prefill queries.
- Reused the original device positions tensor instead of rebuilding it for
  full prefill.
- Enabled the existing SWA prefill GPU kernel for short prompts by default.
- Added `DSV4_DISABLE_COMPRESSED_PREFILL=1` as an A/B and rollback switch.

Verification:

- InfiniCore build: passed.
- InfiniLM build: passed.
- Layer0-4 default: Prefill TTFT 351.46 ms, Decode Avg ITL 9.86 ms,
  Decode Throughput 101.41 tok/s.
- Layer0-4 with compressed prefill disabled: Prefill TTFT 470.20 ms,
  Decode Avg ITL 10.08 ms, Decode Throughput 99.17 tok/s.
- Full model: Prefill TTFT 2838.69 ms, Decode Avg ITL 115.78 ms,
  Decode Throughput 8.64 tok/s.
- Full model produced a coherent Chinese response.

Assessment: accepted. Full-model TTFT improved by 6.79% relative to Stage 1
and 5.29% relative to the original baseline. Decode also improved slightly.

## Stage 3 - Incremental Decode Compressor and Indexer

Status: accepted (incremental compressor); long-context indexer remains future work

Goal: avoid recomputing compressor projections over the complete token history
whenever a new C4 or C128 compressed block becomes visible.

Changes:

- Added an appendable, preallocated compressed-KV cache to
  `DeepseekV4AttentionState`.
- At a C4/C128 block boundary, projected only the newest 8/128-token source
  slice and appended the newly visible compressed block.
- Kept full-history recomputation as a guarded fallback for cache discontinuity.
- Added `DSV4_DISABLE_INCREMENTAL_COMPRESSOR=1` as an A/B and rollback switch.

Verification:

- InfiniLM build: passed.
- Layer0-4 default: Prefill TTFT 351.86 ms, Decode Avg ITL 9.79 ms,
  Decode Throughput 102.10 tok/s.
- Layer0-4 with incremental compressor disabled: Prefill TTFT 330.62 ms,
  Decode Avg ITL 9.78 ms, Decode Throughput 102.24 tok/s.
- Full model default: Prefill TTFT 2859.63 ms, Decode Avg ITL 116.27 ms,
  Decode Throughput 8.60 tok/s.
- Full model with incremental compressor disabled: Prefill TTFT 2841.50 ms,
  Decode Avg ITL 118.08 ms, Decode Throughput 8.47 tok/s.
- Default and rollback paths both produced coherent full-model output; the
  layer0-4 output was identical.

Assessment: accepted based on the same-condition full-model A/B comparison.
Incremental compression improved Decode Avg ITL by 1.53% and throughput by
1.53%. TTFT is unaffected by design and remains within run-to-run noise. The
short benchmark does not enter the long-context indexer path, so that work is
tracked separately rather than credited to this stage.

## Stage 4 - Reuse Device Position IDs in RoPE

Status: accepted

Goal: remove two CPU-to-device position-tensor constructions per attention
layer and generated token, one before Q RoPE and one before K RoPE.

Changes:

- Added a `DeepseekV4RoPE::forward` overload that accepts the engine-owned
  device position tensor.
- Reused contiguous I32/I64 device positions when shape and device match.
- Preserved the vector-based construction as a fallback.
- Added `DSV4_DISABLE_DEVICE_ROPE_POSITIONS=1` as an A/B and rollback switch.

Verification:

- InfiniLM build: passed.
- Layer0-4 default: Prefill TTFT 353.92 ms, Decode Avg ITL 9.29 ms,
  Decode Throughput 107.59 tok/s.
- Layer0-4 with device RoPE positions disabled: Prefill TTFT 348.32 ms,
  Decode Avg ITL 9.75 ms, Decode Throughput 102.61 tok/s.
- Full model default: Prefill TTFT 2826.65 ms, Decode Avg ITL 111.94 ms,
  Decode Throughput 8.93 tok/s.
- Full model with device RoPE positions disabled: Prefill TTFT 2980.94 ms,
  Decode Avg ITL 117.53 ms, Decode Throughput 8.51 tok/s.
- Default and rollback paths produced identical full-model output.

Assessment: accepted. In the same-condition full-model A/B test, TTFT improved
by 5.18%, Decode Avg ITL improved by 4.76%, and decode throughput improved by
4.94%. This confirms that the repeated small host-to-device position transfers
were a material decode synchronization cost.

## Stage 5 - In-place Partial RoPE

Status: accepted

Goal: match SGLang's in-place trailing-dimension RoPE behavior and eliminate
the `narrow -> contiguous -> RoPE -> cat` materialization sequence.

Changes:

- Applied the existing stride-aware InfiniCore RoPE operator directly to the
  trailing 64-dimension view of the normalized Q/K tensor.
- Returned the original contiguous full-head tensor after the in-place update.
- Added `DSV4_DISABLE_INPLACE_PARTIAL_ROPE=1` as an A/B and rollback switch.

Verification:

- InfiniLM build: passed.
- Layer0-4 default: Prefill TTFT 358.99 ms, Decode Avg ITL 9.05 ms,
  Decode Throughput 110.56 tok/s.
- Layer0-4 with in-place partial RoPE disabled: Prefill TTFT 355.08 ms,
  Decode Avg ITL 9.21 ms, Decode Throughput 108.56 tok/s.
- Full model default: Prefill TTFT 2857.57 ms, Decode Avg ITL 103.85 ms,
  Decode Throughput 9.63 tok/s.
- Full model with in-place partial RoPE disabled: Prefill TTFT 2820.92 ms,
  Decode Avg ITL 113.80 ms, Decode Throughput 8.79 tok/s.
- Default and rollback paths produced identical full-model output.

Assessment: accepted. In the same-condition full-model A/B test, Decode Avg
ITL improved by 8.74% and decode throughput improved by 9.56%. TTFT changed by
1.30% in the slower direction, which is within the observed prefill variance
and is not caused by an additional prefill operation. The decode gain comes
from removing two temporary copies and two concatenations per layer/token.

## Rejected Experiment - Imported SGLang Norm+RoPE Wrappers

Status: rejected; not integrated

- `dsv4_sglang_main_q_norm_rope` and `dsv4_sglang_fused_norm_rope` could not
  resolve `libdeepseek_v4_ops.so` through their default lookup.
- With `DEEPSEEK_V4_OPS_SO` set to the repository's absolute library path, both
  standalone Hygon tests triggered HSA invalid-address VMFaults.
- The two generated core dumps were approximately 5.08 GB and 5.05 GB and were
  deleted immediately after diagnosis.
- No model code was changed to call these unstable wrappers. The accepted
  native InfiniCore RMSNorm and stride-aware RoPE path remains active.

## Stage 6 - Prefer the SWA Decode Kernel

Status: accepted

Goal: prevent decode requests with `query_len == 1` from being captured by the
short-prefill path before the specialized SWA decode kernel can run.

Changes:

- Preferred `deepseek_v4_swa_decode` when `query_len == 1` and
  `query_start > 0`.
- Retained `deepseek_v4_swa_prefill` for actual prefill, including one-token
  prefill.
- Added `DSV4_DISABLE_PREFER_SWA_DECODE=1` as an A/B and rollback switch.

Verification:

- InfiniLM build: passed.
- Layer0-4 default: Prefill TTFT 365.62 ms, Decode Avg ITL 8.90 ms,
  Decode Throughput 112.38 tok/s.
- Layer0-4 with preferred decode disabled: Prefill TTFT 359.01 ms,
  Decode Avg ITL 8.95 ms, Decode Throughput 111.74 tok/s.
- Full model default: Prefill TTFT 2844.28 ms, Decode Avg ITL 103.46 ms,
  Decode Throughput 9.67 tok/s.
- Full model with preferred decode disabled: Prefill TTFT 2814.24 ms,
  Decode Avg ITL 104.54 ms, Decode Throughput 9.57 tok/s.
- Default and rollback paths produced identical full-model output.

Assessment: accepted. Full-model Decode Avg ITL improved by 1.03% and decode
throughput improved by 1.04%. The change also removes repeated uploads of the
full SWA key-position vector during decode.

## Stage 7 - Cache the No-index Device Sentinel

Status: accepted

Goal: remove the per-token host-to-device construction of the `{-1}` index
sentinel used by C128 compressed-attention layers without an indexer.

Changes:

- Created one I64 no-index sentinel per compressed-attention layer during model
  construction and reused it throughout prefill/decode.
- Added `DSV4_DISABLE_CACHED_NO_INDEX_SENTINEL=1` as an A/B and rollback switch.

Verification:

- InfiniLM build: passed.
- Layer0-4, 160 output tokens, default: Prefill TTFT 370.35 ms,
  Decode Avg ITL 9.21 ms, Decode Throughput 108.60 tok/s.
- Layer0-4, 160 output tokens, cached sentinel disabled: Prefill TTFT 357.52 ms,
  Decode Avg ITL 9.29 ms, Decode Throughput 107.67 tok/s.
- Full model default: Prefill TTFT 2811.08 ms, Decode Avg ITL 100.38 ms,
  Decode Throughput 9.96 tok/s.
- Full model with cached sentinel disabled: Prefill TTFT 2869.06 ms,
  Decode Avg ITL 109.50 ms, Decode Throughput 9.13 tok/s.
- Default and rollback paths produced identical full-model output.

Assessment: accepted. Full-model Decode Avg ITL improved by 8.33% and decode
throughput improved by 9.09%. The large effect relative to the data size shows
that the eliminated small transfer was acting as a repeated stream
synchronization point across the 20 C128 layers.

## Stage 8 - Precomputed Compressed Block Positions

Status: accepted

Goal: remove repeated C4/C128 block-position vector construction and
host-to-device transfer as compressed blocks become visible during decode.

Changes:

- Precomputed the standard `block_id * compress_ratio` position table on the
  target device during model construction.
- Used a zero-copy `narrow` view for contiguous requests starting at position 0.
- Preserved the dynamic cached-vector path for nonstandard/nonzero positions.
- Added `DSV4_DISABLE_PRECOMPUTED_BLOCK_POSITIONS=1` as an A/B and rollback
  switch.

Verification:

- InfiniLM build: passed.
- Layer0-4, 160 output tokens, default: Prefill TTFT 358.97 ms,
  Decode Avg ITL 9.19 ms, Decode Throughput 108.80 tok/s.
- Layer0-4, 160 output tokens, precomputed positions disabled:
  Prefill TTFT 370.94 ms, Decode Avg ITL 9.28 ms,
  Decode Throughput 107.80 tok/s.
- Full model default: Prefill TTFT 2857.91 ms, Decode Avg ITL 97.80 ms,
  Decode Throughput 10.22 tok/s.
- Full model with precomputed positions disabled: Prefill TTFT 2855.74 ms,
  Decode Avg ITL 100.69 ms, Decode Throughput 9.93 tok/s.
- Default and rollback paths produced identical full-model output.

Assessment: accepted. Full-model Decode Avg ITL improved by 2.87% and decode
throughput improved by 2.92%. TTFT was unchanged within measurement precision.

## Current Result

Compared with the original full-model baseline:

- Prefill TTFT: 2997.23 ms -> 2822.63 ms (5.83% improvement).
- Decode Avg ITL: 117.66 ms -> 98.93 ms (15.92% improvement).
- Decode Throughput: 8.50 tok/s -> 10.11 tok/s (18.94% improvement).
- Full model: loaded successfully with TP=8 and produced the same coherent
  Chinese response in the final default/rollback A/B test.

These are single-run endpoint values. Per-stage attribution uses adjacent
same-condition default/rollback runs; it should not be reconstructed by adding
all percentages because machine noise and interactions between stages are not
additive.

## Stage 9 - Contiguous-position CPU Metadata Fast Path

Status: accepted

Goal: remove repeated host-side scans and temporary position construction from
the per-layer decode path, while fixing the compressed-attention GPU eligibility
check for long contexts.

Changes:

- Tracked whether cached positions are contiguous in `DeepseekV4AttentionState`.
- Replaced repeated sliding-window position scans with O(1) start/length
  calculation for standard contiguous decode requests.
- Reused the input device position tensor for both query and key positions in
  SWA prefill when the request starts at position zero.
- Based compressed-attention GPU eligibility on the effective selected-key
  count (`index_top_k`) rather than all available compressed blocks.
- Added `DSV4_DISABLE_CONTIGUOUS_WINDOW_FASTPATH=1`,
  `DSV4_DISABLE_SWA_PREFILL_POSITION_REUSE=1`, and
  `DSV4_DISABLE_EFFECTIVE_COMPRESSED_KEY_LIMIT=1` rollback switches.

Verification:

- InfiniLM build: passed.
- Layer0-4 default: Prefill TTFT 338.15 ms, Decode Avg ITL 9.12 ms,
  Decode Throughput 109.71 tok/s.
- Layer0-4 with the contiguous-window and prefill-position fast paths disabled:
  Prefill TTFT 358.09 ms, Decode Avg ITL 9.15 ms,
  Decode Throughput 109.30 tok/s.
- Full model default: Prefill TTFT 2822.63 ms, Decode Avg ITL 98.93 ms,
  Decode Throughput 10.11 tok/s.
- Full model with the contiguous-window and prefill-position fast paths
  disabled: Prefill TTFT 2872.34 ms, Decode Avg ITL 109.59 ms,
  Decode Throughput 9.12 tok/s.
- Default and rollback paths produced identical full-model output.

Assessment: accepted. In the adjacent full-model A/B, Prefill TTFT improved by
1.73%, Decode Avg ITL improved by 9.73%, and decode throughput improved by
10.86%. The effective compressed-key limit is a long-context correctness and
GPU-path fix; the 160-token benchmark does not exercise that boundary.

## Rejected Follow-up Experiments

### Query-only residual cache

The decode state was changed temporarily to retain only the current
`q_residual` token instead of copying its full history. Shapes and output were
correct, but the full-model result regressed from 97.75 ms / 10.23 tok/s on the
old cache to 98.92 ms / 10.11 tok/s on the query-only cache. The change was
fully reverted.

### Compressed attention probability cache

Two versions precomputed softmax probabilities in shared memory inside
`deepseek_v4_compressed_decode`. The first added a block synchronization and
regressed full-model Decode Avg ITL from 99.00 ms to 99.83 ms. The second
removed that synchronization but still regressed layer0-4 Decode Avg ITL from
9.21 ms to 9.25 ms. Both versions were fully reverted; InfiniCore has no diff
from these experiments.

## Stage 10 - GPU-only Decode Paths and Attention Structure Cleanup

Status: accepted as structural cleanup; performance neutral within measurement
noise

Goal: make the production GPU execution paths explicit, remove unreachable or
unsafe CPU decode fallbacks, and align function names with their actual roles.

Changes:

- Renamed the main paths to `compressed_attention_gpu_`,
  `sliding_attention_gpu_`, and `attention_prefill_`.
- Removed the CPU sliding-attention fallback and the CPU compressed-decode
  fallback. Unsupported device, position, or key-count combinations now fail
  explicitly instead of silently copying tensors to the host.
- Removed obsolete rollback environment switches associated with the deleted
  CPU paths.
- Removed the GPU Indexer try/catch fallback. GPU execution now calls the fused
  compressor/indexer path directly; the explicit CPU-device reference path is
  retained.
- Removed unused position/debug utility functions and their declarations.
- Kept the full prefill CPU reference for explicit CPU execution and
  non-contiguous-position compatibility. It is not used by the standard Hygon
  paged-attention path.
- Changed the InfiniCore compressed-decode descriptor limit to use the
  effective selected compressed-key count (`index_top_k`) rather than all
  available compressed blocks. The no-Indexer path still validates the full
  compressed-block count.

Verification:

- InfiniCore Hygon build and install: passed.
- InfiniLM Hygon build and install: passed.
- `git diff --check`: passed.
- Layer0-4, 12 input / 160 output tokens: output was identical before and after
  the cleanup.
- Adjacent same-machine A/B: old Decode Avg ITL 9.21 ms / 108.54 tok/s; new
  Decode Avg ITL 9.27 ms / 107.91 tok/s. The 0.65% difference is within the
  observed run-to-run variance and no speedup is attributed to this stage.
- Layer0-4, 2050 input / 8 output tokens: passed. This crosses the C4 full
  coverage threshold and exercises the fused GPU Indexer, compressed KV cache,
  and compressed-decode path without a CPU fallback.
- Full 43-layer model, TP=8: passed and produced the same coherent Chinese
  response. Two post-change runs measured 2857.06 / 99.70 ms / 10.03 tok/s and
  2835.75 / 101.93 ms / 9.81 tok/s for TTFT / Decode Avg ITL / throughput.

Assessment: accepted. This stage reduces the attention implementation by about
485 lines and makes decode behavior deterministic: supported Hygon requests
remain on GPU, while unsupported cases report a clear error. The remaining CPU
reference is isolated to `attention_prefill_` and can be removed separately
after CPU and non-contiguous-position support are formally retired.

## Stage 11 - Remove CPU Attention Reference Paths

Status: accepted

Goal: remove CPU numerical implementations that are not used by the supported
Hygon paged-attention path, and make unsupported execution modes explicit.

Changes:

- Removed the full CPU attention implementation from `attention_prefill_`,
  including host dot products, masking, softmax, compressed-block attention,
  inverse RoPE, and the final host-to-device result copy.
- GPU prefill still dispatches to `sliding_attention_gpu_` or
  `compressed_attention_gpu_`. CPU tensors and non-contiguous positions now
  report an explicit unsupported-path error.
- Removed `DeepseekV4Compressor::forward_values`, its CPU RMSNorm/pooling
  helpers, host weight caches, and the unused generic `forward` wrapper.
- Removed the Indexer CPU score/top-k implementation and host-vector `forward`
  wrapper. `forward_tensor` now requires a GPU tensor and dispatches directly
  to `infinicore::op::deepseek_v4_indexer`.
- Removed the block-level and inverse CPU RoPE methods that only served the
  deleted attention reference path.
- Kept the one-time APE weight-layout conversion in
  `process_weights_after_loading`; the current GPU compressor kernel requires
  the converted layout.

Verification:

- Net implementation change: 30 insertions, 435 deletions across seven files.
- InfiniLM Hygon build and install: passed.
- Layer0-4, 12 input / 160 output tokens: output identical. Before cleanup:
  TTFT 368.74 ms, Decode Avg ITL 9.28 ms, 107.81 tok/s. After cleanup:
  TTFT 342.56 ms, Decode Avg ITL 9.14 ms, 109.45 tok/s.
- Layer0-4, 2050 input / 8 output tokens: passed, exercising the C4 GPU
  Indexer and compressed-decode path. TTFT 8723.08 ms, Decode Avg ITL
  694.42 ms, 1.44 tok/s.
- Full 43-layer model, TP=8: loaded successfully and produced the same coherent
  Chinese response. TTFT 2806.95 ms, Decode Avg ITL 97.48 ms, 10.26 tok/s.

Assessment: accepted. The deleted implementations were cold compatibility
paths, so the apparent benchmark improvements are treated as machine variance
rather than attributed performance gains. The main benefit is a smaller and
deterministic GPU-only attention implementation. Remaining host position
metadata and Indexer position transfers are separate performance work.

## Stage 12 - Homogeneous Paged Batch Support

Status: accepted

Goal: restore paged-attention prefill and decode for homogeneous batches after
the GPU-only prefill cleanup exposed the packed input layout.

Root cause:

- The paged engine packs a logical `[batch, sequence, hidden]` tensor as
  `[1, batch * sequence, hidden]` and supplies one repeated position segment
  per request.
- `DeepseekV4Attention::forward_paged_` treated the packed token dimension as
  one sequence and required its repeated positions to be globally contiguous.
  CSA/HCA layers therefore failed during full-model batch prefill with
  `prefill requires a GPU device and contiguous positions`.
- The layer0 model did not expose the failure because it did not exercise the
  same CSA/HCA compressed-attention path.

Changes:

- Recover the logical batch size from `AttentionMetadata::input_offsets`, view
  the packed hidden states as `[batch, sequence, hidden]`, and repack the
  projected output before returning to the engine.
- Validate that the current paged batch is homogeneous: requests must have the
  same sequence length and identical position segments. Reuse the first device
  position segment for RoPE and fused attention kernels.
- Preserve batch dimensions in incremental compressed-KV storage instead of
  flattening cached blocks through an implicit batch size of one.

Verification:

- InfiniLM Hygon build and install: passed.
- Full 43-layer model, TP=8, batch=4 warmup at input length 128 plus five decode
  steps: passed.
- Full model, TP=8, batch=4, 12 input / 16 output tokens: TTFT 387.70 ms,
  Decode Avg ITL 107.97 ms, aggregate decode throughput 37.05 tok/s.
- Full model, TP=8, batch=1 regression, 12 input / 16 output tokens: TTFT
  157.93 ms, Decode Avg ITL 98.85 ms, decode throughput 10.12 tok/s.
- Both batch sizes produced the same coherent prefix: `你好！我是DeepSeek，一个由深度公司研发的AI助手，`.
- `git diff --check` passed, no benchmark process remained, and no core file
  larger than 100 MB was generated under `/workspace_infini` or `/workspace`.

Assessment: accepted. The previously failing homogeneous batch path now runs
entirely through the existing GPU CSA/HCA kernels. Variable-length paged
batches remain explicitly unsupported because the current attention state and
compressed-decode interfaces use one shared logical position vector.

## Stage 13 - C4 Indexer Long-Context Optimization

Status: accepted; the 7168-output full-model run remains limited by GPU memory
on the current machine

Goal: remove the severe C4 Indexer performance cliff after the compressed
sequence exceeds `index_topk=512`, then measure batch 4 / input 1024 / output
7168.

Root cause:

- The original Indexer kernel launched only `batch * query_len` GPU blocks. For
  every one of the 512 TopK selections it rescanned all compressed blocks and
  recomputed all 64-head, 128-dimensional dot products.
- `DeepseekV4Indexer::forward_tensor` also recompressed the complete hidden
  history on every decode step after the full-coverage shortcut ended.
- On layer0-4, the first sparse step at 513 compressed blocks took about
  1062 ms, versus an 18.18 ms normal decode step. Eight-device utilization was
  approximately 5%.

Changes:

- Split the InfiniCore Indexer into a parallel score kernel and a TopK selection
  kernel. Scores are computed once over `(batch, query, compressed_block)`;
  selected scores are invalidated in workspace instead of rescoring or scanning
  an `already_selected` list.
- Added an incremental C4 compressed-key cache to `DeepseekV4Indexer`. The first
  sparse call creates the full cache; later calls reuse it and compress only the
  most recent eight tokens when one new C4 block becomes available.
- Attached the cache to the recursive module runtime-reset lifecycle so it is
  cleared between requests and benchmark cases.

Verification:

- InfiniCore Hygon build and install: passed.
- InfiniLM Hygon build and install: passed.
- `git diff --check`: passed in both repositories.
- Layer0-4, TP=8, batch=4, input 1024, output 1024 before the sparse boundary:
  TTFT 1267.87 ms, Decode Avg ITL 18.18 ms, 220.07 tok/s.
- Layer0-4, output 1029, first sparse step: total generation improved from
  20997.01 ms before the new kernel to 20133.98 ms after it. The estimated
  sparse-step cost fell from about 1062 ms to about 199 ms.
- Layer0-4, output 1100 with incremental cache active: TTFT 1266.17 ms,
  Decode Avg ITL 18.61 ms, 214.89 tok/s.
- Layer0-4 target case, output 7168: completed in 176797.58 ms. TTFT was
  1264.25 ms, Decode Avg ITL 24.49 ms, and throughput was 163.32 tok/s. The
  pre-change run had not completed after more than 20 minutes.
- Full 43-layer model, TP=8, batch=4, input 1024, output 3073: completed and
  produced coherent Chinese output. TTFT was 13283.94 ms, Decode Avg ITL was
  193.24 ms, throughput was 20.70 tok/s, and generation took 606915.06 ms.
- On this machine, output 3073 is the maximum safe value for this batch/input
  configuration. It leaves the attention history at capacity 4096. The output
  7168 run failed at approximately the same 605-second point when the next
  token required history capacity to grow from 4096 to 8192; HYGON:0 reported
  a device allocation failure after memory reached 97%.
- A direct boundary test with output 3074 reproduced the allocation failure on
  HYGON:0 after 10 minutes 17 seconds. This is the first output length whose
  final decode step requires logical history 4097 and therefore confirms 3073
  as the measured maximum on this machine, rather than only an inferred limit.
- No benchmark process or core file larger than 100 MB remained after testing.

Assessment: accepted. The algorithmic Indexer cliff is removed: layer0-4 now
completes the requested 7168-output case in under three minutes. Completing the
same case on the current full-model machine requires bounded hidden-state
storage or more GPU memory. On a larger-memory machine, rerun the saved full
case directly and record TTFT, Decode Avg ITL, throughput, and peak memory.
