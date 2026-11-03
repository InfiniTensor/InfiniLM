# Ascend lossless shared-KV projection regression

`shared_kv_projection.cc` exports `check_shared_kv_projection(int variant)`.
Compile this as a shared library against the installed InfiniLM extension and
InfiniCore C++ API. Run each variant in its own process with four visible Ascend
devices, `INFINILM_ASCEND_SHARE_KV_PROJECTION=1`,
`INFINICORE_ASCEND_REUSE_NZ_EXECUTOR=1` and
`INFINICORE_GRAPH_SYNC_WARMUP=1`. The fixture configures TP4 and pre-transposed
BF16 weights itself. Load the exported symbol with ctypes and require return 0.

Variants are exact repeated K/V heads with bias (0), one changed K weight (1),
one changed K bias (2), and exact repeated heads without bias (3). Variants 1
and 2 must retain the original projection. All variants compare complete graph
outputs against the original projection and sampled CPU dot products for
1, 16, 17, 41, 265, 1024, 2047 and 2048 tokens, each with five changing inputs.
The test assembles Q/K/V using graph-aware slice copies; eager `cat` is not a
valid graph-recording reference in this backend.

The optimization is opt-in and defaults off. It compares every byte of K/V
weights and bias while loading, retains the original parameter tensors, and
builds a private packed weight only for exactly repeated contiguous heads.
Inference projects each distinct K/V head once and restores all original MHA
heads before bias, RoPE, cache writes and attention. It does not change the
checkpoint, head count, model config or KV-cache layout. Shapes outside its
validated prefill range (17 through 2048 tokens) use the original path.

Changing GEMM output width can change accumulation/rounding on a new backend
or shape. Before accepting a performance result, compare identical teacher-
forced inputs on the real four-layer TP4 model and the full 80-layer TP4 model.
The latter must cover batch 16 across a 256-token page boundary plus batches
4, 1 and 8. Unit output equality alone is insufficient. Then run full-length
service benchmarks and a separate natural-language check.
