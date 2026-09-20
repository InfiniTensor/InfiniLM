# Mamba2 NVIDIA Adapter Report

## Scope

- Model: `state-spaces/mamba2-130m`
- Platform: NVIDIA RTX 4090 D, 24 GB
- Runtime: InfiniLM + InfiniCore CUDA backend
- Supported path: single GPU, `num_groups=1`, paged scheduler with recurrent state rows
- Graph compilation: disabled for this adapter
- Tensor parallel: not implemented

## Implementation

The adapter implements the Mamba2 path in the existing InfiniLM model registry:

1. Input projection is split into gate, SSM input, B/C parameters, and time-step values.
2. The existing CUDA causal-convolution operator updates one convolution history per request.
3. The existing CUDA selective-scan operator updates the recurrent SSM state.
4. RMSNorm and the SiLU gate are applied before the output projection.
5. Each request owns a state row, so requests do not share convolution or SSM history.
6. Mamba2 requests use only recurrent state rows; no Transformer attention KV
   cache is needed by this pure recurrent architecture.

The existing `BlockManager` and its KV admission calculations were left
unchanged. This adapter distinguishes a pure recurrent model from a hybrid
model that has both recurrent state and attention KV; it does not claim to fix
generic KV over-reservation.

## Problems Found And Fixes

### Build environment

- The minimal remote xmake package had no Python module rule. The target is built as a shared library named `_infinilm.so` instead.
- The remote build needed the Python 3.12 include directory and `INFINI_ROOT=/data/InfiniCore/install`.
- Root xmake execution requires `xmake --root`.
- The first full rebuild used the default 130 parallel jobs and one compiler was killed by memory pressure. The successful rebuild used four jobs and one link job.

### Test reference

- The first PyTorch reference used the normalized tensor as the residual. The implementation correctly uses the original input as the residual.
- The reference did not write convolution and SSM states back to the request state table, hiding decode-state errors. Both state tables now persist after every forward.

### Official checkpoint configuration

- The released config omits `model_type`, Mamba2 dimensions, and the padded vocabulary size. A preparation script adds these fields.
- The embedding weight is `50288 x 768` although the original config says `50277`; the prepared config uses `50288`.
- The released projection and convolution shapes identify `state_size=128` for this checkpoint. The prepared config records that value.
- The official `mixer.norm.weight` has 1536 values, so the adapter supports a configurable full-intermediate RMSNorm in addition to the tiny-test head-dimension form.

## Verification

### Unit and correctness tests

- Mamba2 checkpoint/config and weight-remapping tests: passed, 4 tests.
- Mamba2 scheduler state-cache test: passed, 1 test.
- Tiny random-weight GPU reference test: passed. It covers prefill, decode, recurrent state persistence, and request isolation with `atol=rtol=2e-4`.
- Real Mamba2-130M load/prefill/decode test: passed.

### Real-model benchmark

Input length is 128, generated length is 32, five measured runs after two warmups.

| Batch | Prefill time | Decode time | Decode throughput | GPU used after benchmark |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 145.4 ms | 158.3 ms | 202.1 tok/s | 1385 MiB |
| 4 | 577.1 ms | 322.5 ms | 396.9 tok/s | 1393 MiB |

The observed GPU usage after loading was about 1307 MiB out of 24564 MiB. The value includes the process/runtime baseline reported by `nvidia-smi`, so it is not a pure parameter-size measurement.

## Current Limitations

- NVIDIA CUDA only; no Ascend or other accelerator implementation is included.
- Only `num_groups=1` is supported by the current scan integration.
- Tensor parallel and graph compilation are disabled for Mamba2.
- The official checkpoint repository does not include tokenizer files, so the verified real-model path uses raw token IDs. Text tokenizer and chat-generation verification need a compatible GPT-style tokenizer directory.
- The benchmark is a small raw-token benchmark, not a full service concurrency evaluation.
