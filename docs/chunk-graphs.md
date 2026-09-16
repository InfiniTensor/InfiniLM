# Experimental graphs with chunked Prefill

Supported chunk/graph combinations are TP1/PP1 and TP2/PP1 on CUDA-compatible devices. Tested backends are NVIDIA A6000 paged attention (Qwen2.5-1.5B FP16) and MetaX C500 Flash Attention (Qwen3 BF16, TP1). PP2 chunking uses eager execution.

Set `prefill_chunk_size=512`, `enable_graph=True` and `device="cuda"` to combine eager Prefill with Decode graphs. Use `attn_backend="paged-attn"` on the tested A6000 build and `"flash-attn"` on C500. InfiniCore must be built with `--graph=y`; TP2 also requires working collectives. The `cuda` device alias maps to MACA in the C500 build.

To additionally test fixed-size Prefill graphs, set
`INFINILM_PREFILL_GRAPH_CHUNK_SIZE=512` **before creating the engine**. This
experimental switch requires the MetaX backend, TP1, ordinary paged KV and
Flash Attention. It captures one request of exactly the chosen size; other
sizes retain the existing execution path. Single-token final tails may use
the Decode graph because they have the same one-query attention semantics.

The compiler captures separate intermediate and final graphs. Intermediate
chunks execute every Transformer layer but omit the LM head, sampling and
token output, matching the eager implementation. A graph with no logits is
still a completed execution and must not trigger a duplicate eager forward.

Token IDs, positions, KV lengths, offsets, page tables and slot mappings are
updated in persistent buffers before each replay. The scalar maximum KV
length is fixed to the cache pool capacity so that later chunks and reused
prefixes do not inherit the first chunk's length. Unsupported shapes fall
back to the existing execution path. This implements fixed shapes, without
padding or dynamic Prefill batch buckets.

The integrated cache manager supports LRU and optional SLRU with both graph modes. NVIDIA TP2 Decode graphs and PP2 eager have lifecycle validation; fixed-size Prefill graphs remain MetaX TP1 only. Remote KV, PP graphs, speculative decoding and hybrid/MoE/multimodal models are outside this scope.

A C500 short ablation (Qwen3-4B, 2048 input/16 output, chunk512) measured single-request token intervals of 12.96 ms for chunk eager and 9.37 ms with Decode graphs. Adding Prefill graphs reduced TTFT from 247.97 to 241.37 ms. These samples precede the cache-policy integration and do not establish sustained serving throughput. Graphs add initialization cost and retained workspace; full additional memory usage was not measured.

Integrated lifecycle results and limits: [cache-chunk-validation.md](cache-chunk-validation.md).
