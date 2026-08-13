#pragma once

#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

#include <cstdint>
#include <optional>

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(RotaryEmbedding, const Tensor &, Tensor, std::optional<Tensor>, const Tensor &, int64_t, bool, int64_t, bool);

void rotary_embedding_(const Tensor &positions,
                       Tensor query,
                       std::optional<Tensor> key,
                       const Tensor &cos_sin_cache,
                       int64_t head_size,
                       bool is_neox,
                       int64_t rope_dim_offset = 0,
                       bool inverse = false);

} // namespace infinicore::op
