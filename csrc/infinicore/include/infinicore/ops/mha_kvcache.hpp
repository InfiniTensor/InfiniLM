#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"
#include <optional>

namespace infinicore::op {

// Flash Attention KV-cache decode op.
//
// Wraps flash::mha_fwd_kvcache for single-step (decode) attention over a
// paged KV cache.
//
// Tensor shapes:
//   out         : [batch_size, seqlen_q, num_heads, head_size]
//   q           : [batch_size, seqlen_q, num_heads, head_size]
//   k_cache     : [num_blocks, block_size, num_heads_k, head_size]   (paged layout)
//   v_cache     : [num_blocks, block_size, num_heads_k, head_size]   (paged layout)
//   seqlens_k   : [batch_size]   int32 — total KV length per request
//   block_table : [batch_size, max_num_blocks_per_seq]  int32

INFINICORE_GRAPH_OP_CLASS(
    MhaKVCache,
    Tensor,                // out
    const Tensor &,        // q
    const Tensor &,        // k_cache
    const Tensor &,        // v_cache
    const Tensor &,        // seqlens_k
    const Tensor &,        // block_table
    std::optional<Tensor>, // alibi_slopes
    float);                // scale

Tensor mha_kvcache(const Tensor &q,
                   const Tensor &k_cache,
                   const Tensor &v_cache,
                   const Tensor &seqlens_k,
                   const Tensor &block_table,
                   std::optional<Tensor> alibi_slopes,
                   float scale);

void mha_kvcache_(Tensor out,
                  const Tensor &q,
                  const Tensor &k_cache,
                  const Tensor &v_cache,
                  const Tensor &seqlens_k,
                  const Tensor &block_table,
                  std::optional<Tensor> alibi_slopes,
                  float scale);

} // namespace infinicore::op
