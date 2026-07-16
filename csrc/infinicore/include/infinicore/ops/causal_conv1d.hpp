#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"
#include <optional>

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(CausalConv1d,
                          Tensor,
                          Tensor,
                          std::optional<Tensor>,
                          const Tensor &,
                          const Tensor &,
                          std::optional<Tensor>,
                          std::optional<Tensor>,
                          std::optional<Tensor>,
                          std::optional<Tensor>);

Tensor causal_conv1d(const Tensor &qkv,
                     Tensor conv_state,
                     const Tensor &weight,
                     std::optional<Tensor> bias = std::nullopt,
                     std::optional<Tensor> cu_seqlens = std::nullopt,
                     std::optional<Tensor> initial_state_indices = std::nullopt,
                     std::optional<Tensor> final_state_indices = std::nullopt);

void causal_conv1d_(Tensor out,
                    Tensor conv_state,
                    std::optional<Tensor> final_conv_state,
                    const Tensor &qkv,
                    const Tensor &weight,
                    std::optional<Tensor> bias,
                    std::optional<Tensor> cu_seqlens,
                    std::optional<Tensor> initial_state_indices,
                    std::optional<Tensor> final_state_indices);

} // namespace infinicore::op
