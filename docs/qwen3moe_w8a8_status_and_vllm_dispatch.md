# Qwen3-MoE W8A8 InfiniLM/vLLM Status

Date: 2026-07-10

## Remote Environment

- Host: `qinyiqun@10.211.3.28`
- SSH key: `C:\Users\qinyi\.ssh\bw1000`
- Container: `qinyiqun`
- InfiniCore: `/home/qinyiqun/InfiniCore`
- InfiniLM: `/home/qinyiqun/InfiniLM`
- FP model: `/home_aclsylqidf/shared/Qwen3-30B-A3B`
- W8A8 model: `/home_aclsylqidf/shared/Qwen3-30B-A3B-Channel-INT8-w8a8`

Runtime setup inside the container:

```bash
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export PATH=/root/.local/bin:/opt/dtk/cuda/cuda/bin:$PATH
export XMAKE_ROOT=y
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:/root/.infini/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/usr/local/:${PYTHONPATH:-}
```

InfiniCore configure must include `--graph=y`:

```bash
xmake f --hygon-dcu=true --aten=true --flash-attn=/usr/local/lib/python3.10/dist-packages/ --cuda=/opt/dtk/cuda/cuda --ccl=true --graph=y -cv -y
xmake build
xmake install
xmake build _infinicore
xmake install _infinicore
pip install -e .
```

## Current Benchmark Contract

- Model family: `Qwen3-30B-A3B`
- Target path: W8A8 quantized model
- Parallelism: `TP=2`, `DP=1`, `EP=1`
- MoE communication: TP only, no DeepEP/allgather EP path
- Benchmark length going forward: `input_len=4096`, `output_len=1280`
- Important guardrail: pass only one `input_len` value. A comma-separated input length list can hang the current benchmark path.
- Device/profiling tools: `hy-smi` and Hygon trace.

## Current InfiniLM Findings

The long-run stall was isolated to the W8A8 MoE path with long prefill. FP graph runs and short W8A8 decode runs can complete, so the issue is not simply long output length.

Observed behavior before long-prefill slicing:

- W8A8 `4096/128` graph timed out.
- W8A8 `4096/128` no-graph segfaulted.
- Backtraces showed one rank waiting in `RankWorker::wait`, while the other rank was inside the W8A8 Marlin MoE path and teardown/exit handling.

The Hygon W8A8 Marlin MoE path now chunks long prefill internally:

- Files: `csrc/layers/moe/runner/cuda_fused_moe_runner.cpp`, `.hpp`
- Fixed chunk size: `16384` tokens, matching vLLM's production chunk size
- The sliced path is selected before full-input routing metadata is prepared, so each token is aligned only once
- No W8A8 slice or debug environment switches are required

This keeps long-prefill workspace bounded while decode continues to use the graph-captured Marlin path directly.

## vLLM W8A8 MoE Path

vLLM package path:

- `/usr/local/lib/python3.10/dist-packages/vllm`
- Runtime version in logs: `v0.15.1`

Important vLLM env:

- `VLLM_FUSED_MOE_CHUNK_SIZE=16384`
- `VLLM_W8A8_BACKEND=3`

Main call chain:

1. `CompressedTensorsW8A8Int8MoEMethod.apply()`
2. `fused_experts(...)`
3. `lmslim.layers.fused_moe.fuse_moe_int8.fused_experts_impl_int8`

vLLM does not repack Qwen3 MoE weights into the InfiniLM Marlin layout. It keeps ordinary channel-wise int8 tensors:

- `w1`: `[E, 768, 2048]`
- `w2`: `[E, 2048, 384]`
- `w1_scale`: `[E, 768, 1]`
- `w2_scale`: `[E, 2048, 1]`
- `E=128`, `top_k=8`

Operator sequence per chunk:

1. Per-token quantize hidden states.
2. Align/count/sort tokens by expert.
3. GEMM1: `lightop.moe_gemm_w8a8(...)`
4. Activation and quantize: `fuse_silu_mul_quant(...)`
5. GEMM2: `lightop.moe_gemm_w8a8(...)`
6. Reduce top-k outputs: `moe_sum` / `moe_reduce_dispatch`

## Representative vLLM Size Dispatch

These are the useful anchor cases for InfiniLM implementation. We do not need to reproduce every tiny graph-capture size immediately.

| Effective M | GEMM1 shape | GEMM1 config/kernel | GEMM2 shape | GEMM2 config/kernel | Notes |
| --- | --- | --- | --- | --- | --- |
| `1..32` | `N=768,K=2048` | small-M `lightop.moe_gemm_w8a8`, often `BLOCK_M=16` | `N=2048,K=384` | small-M `lightop.moe_gemm_w8a8` | decode/graph capture sizes |
| `896` | `N=768,K=2048` | `BLOCK_M=64, MODE=517, DELTA=1`, HIP NT prefill up | `N=2048,K=384` | `BLOCK_M=32, MODE=568, DELTA=2`, HIP NT prefill down | tail chunk |
| `4096` | `N=768,K=2048` | `BLOCK_M=128, MODE=1000, DELTA=1`, `MOE_W8A8_I8_PERCHANNEL_ASM_TN_MT128x256x128_WGM1_UP` | `N=2048,K=384` | `BLOCK_M=64, MODE=517, DELTA=2`, HIP NT prefill down | target single request prefill |
| `10240` | `N=768,K=2048` | `BLOCK_M=128, MODE=1000, DELTA=1`, same ASM UP kernel | `N=2048,K=384` | `BLOCK_M=64, MODE=523, DELTA=2`, HIP NT prefill down | vLLM chunked prefill example |

vLLM with 16 concurrent 8K prompts enabled chunked prefill with `max_num_batched_tokens=10240`. The observed MoE effective sizes were `10240`, `8256`, `896`, plus small graph-capture sizes. This means scheduler chunking, not only `VLLM_FUSED_MOE_CHUNK_SIZE`, controls the actual large-M MoE calls.

## vLLM W8A8 Dense Linear Path

Main call chain:

1. `CompressedTensorsW8A8Int8.apply_weights()`
2. `apply_int8_linear(..., w8a8_strategy=3)`
3. `per_token_quant_int8(...)`
4. `ops.blaslt_scaled_mm(...)`
5. backend 3: `hipblaslt_w8a8_channelwise_gemm`

Representative kernels:

- `M=1,N=4096,K=2048`: small `Cijk_Alik_Bljk_I8BS_MT64x16x256...`
- `M=4096,N=4096,K=2048`: large `Cijk_Alik_Bljk_I8BS_MT256x256x128...`

## Implementation Direction

The next code change should move InfiniLM W8A8 MoE toward vLLM's ordinary channel-wise path:

1. Add a new W8A8 channel MoE backend in InfiniLM, keeping `[E,N,K]` weights and `[E,N,1]` scales instead of calling `moe_w8a8_marlin_pack`.
2. Add/route an InfiniCore wrapper around ordinary `lightop.moe_gemm_w8a8`, not the current `moe_gemm_marlin_w8a8` adaptor.
3. Reuse the existing MoE workspace pattern where possible: int8 hidden cache, int8 intermediate cache, per-token scales, BF16 intermediate/output buffers.
4. Select configs by effective `M` and GEMM shape, matching the vLLM anchors above first: small decode, `896`, `4096`, `10240`.
5. Default the MoE chunk cap to `16384` for the ordinary channel path, matching vLLM's fused MoE chunk cap. Scheduler-level chunking is still needed later for 16 concurrency and 8K-10K contexts.

## Useful Remote Artifacts

- vLLM probe log: `/tmp/vllm_w8a8_kernel_probe_i8192_c16_o16_20260710_110631.server.log`
- MoE micro traces:
  - `/tmp/hygon_trace_lmslim_w8a8_moe_m10240_20260710_111507`
  - `/tmp/hygon_trace_lmslim_w8a8_moe_m896_20260710_111603`
  - `/tmp/hygon_trace_lmslim_w8a8_moe_m4096_20260710_111646`
- Dense linear micro traces:
  - `/tmp/hygon_trace_vllm_w8a8_linear_m1_n4096_k2048_20260710_113331`
  - `/tmp/hygon_trace_vllm_w8a8_linear_m4096_n4096_k2048_20260710_113413`
