# Hygon VideoNSA fresh video prefill

The paged eager worker supports a single fresh, complete video request (2..8192
prompt tokens, no cached prefix). Python freezes CPU media tensors; the native
worker copies and checks them again before any device upload, embedding
replacement or raw KV write. Grids, pixel counts, replacement bounds, request IDs,
visual ranges and all three MRoPE axes must agree. Unsupported partial/multi-request
media input is rejected. Ordinary text remains on the existing prefill path.

The full native prefill uses the model's top4, block64 and window256 with its
three gate projections. The frozen standalone benchmark's top32 is independent.
Paged raw storage remains BSHD; decode uses a BHSD strided view. Fresh prefill
revokes compressed-cache evidence; the first decode rebuilds compression before
later one-tail updates. Runtime, pool epoch, invocation and completion checks
remain in force. Borrowed KV writers remain unsupported.

Model Graph, speculative inputs, KV connectors, multimodal cached prefixes,
chunked media and multiple simultaneous video requests are outside this domain.
Operator-level Graph checks do not imply model Graph support.

Tests and evidence are in the personal workspace's
`evidence/video-prefill-integration/`. Diagnostic worker CPU copies are produced
only by a separate personal diagnostic build, and are absent from ordinary source.

## Vision precision alignment on BW

VideoNSA vision blocks use RMSNorm. The separately added LayerNorm provider is
not the active normalization on this model path.

Unquantized biased BF16 vision projections on Hygon add bias in FP32 before the
final BF16 cast, preserving the historical single-rounding contract. Other model
paths retain the shared dense policy. The helper accepts replicated Linear only
and preserves explicit output dtype, metadata validation and fallback behavior.

For the measured short-video M=256, K=1280, N=3420 vision-MLP shape, native BLAS
with BF16 operands and FP32 output showed different sums for identical inputs.
Only this shape uses exact BF16-to-FP32 operand casts before native GEMM. Bias
is still added before the final BF16 cast. This policy is local to VideoNSA;
common GEMM and dense implementations are unchanged.

`evidence/precision-alignment/` records the production-helper tests, 24-repeat
small-projection test, ordinary full-weight short/long runs (two requests each,
eight generated tokens matching the historical valid version), and a separate
16-forward diagnostic state audit. Diagnostic source is removed and the
ordinary library is rebuilt. The historical logits/KV are not bitwise identical;
these checks do not establish general model-quality or speed acceptance.
