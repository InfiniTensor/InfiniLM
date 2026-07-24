#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(NsaPagedAttention, Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, float, int, int, int);

Tensor nsa_paged_attention(const Tensor &q, const Tensor &k_cmp, const Tensor &v_cmp, const Tensor &k_cache, const Tensor &v_cache,
                           const Tensor &block_tables, const Tensor &kv_lens, const Tensor &gates,
                           float scale, int nsa_block_size, int window_size, int select_blocks);

void nsa_paged_attention_(Tensor out, const Tensor &q, const Tensor &k_cmp, const Tensor &v_cmp, const Tensor &k_cache, const Tensor &v_cache,
                          const Tensor &block_tables, const Tensor &kv_lens, const Tensor &gates,
                          float scale, int nsa_block_size, int window_size, int select_blocks);

} // namespace infinicore::op
