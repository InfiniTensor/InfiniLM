#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"
#include <optional>

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(ChunkGatedDeltaRule,
                          Tensor,
                          Tensor,
                          std::optional<Tensor>,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          std::optional<Tensor>,
                          std::optional<Tensor>,
                          std::optional<Tensor>,
                          bool,
                          size_t);

Tensor chunk_gated_delta_rule(const Tensor &q,
                              const Tensor &k,
                              const Tensor &v,
                              const Tensor &g,
                              const Tensor &beta,
                              Tensor initial_state,
                              std::optional<Tensor> cu_seqlens = std::nullopt,
                              std::optional<Tensor> initial_state_indices = std::nullopt,
                              std::optional<Tensor> final_state_indices = std::nullopt,
                              bool use_qk_l2norm = false,
                              size_t chunk_size = 64);

void chunk_gated_delta_rule_(Tensor out,
                             Tensor initial_state,
                             std::optional<Tensor> final_state,
                             const Tensor &q,
                             const Tensor &k,
                             const Tensor &v,
                             const Tensor &g,
                             const Tensor &beta,
                             std::optional<Tensor> cu_seqlens,
                             std::optional<Tensor> initial_state_indices,
                             std::optional<Tensor> final_state_indices,
                             bool use_qk_l2norm = false,
                             size_t chunk_size = 64);

} // namespace infinicore::op
