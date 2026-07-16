#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"
#include <optional>

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(PagedAttention, Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, std::optional<Tensor>, float);

Tensor paged_attention(const Tensor &q, const Tensor &k_cache, const Tensor &v_cache,
                       const Tensor &block_tables, const Tensor &kv_lens,
                       std::optional<Tensor> alibi_slopes, float scale);

void paged_attention_(Tensor out, const Tensor &q, const Tensor &k_cache, const Tensor &v_cache,
                      const Tensor &block_tables, const Tensor &kv_lens,
                      std::optional<Tensor> alibi_slopes, float scale);

} // namespace infinicore::op
