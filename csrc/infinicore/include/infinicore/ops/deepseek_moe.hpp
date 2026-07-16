#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"
#include <vector>

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(
    DeepseekMoe,
    Tensor,
    const Tensor &,
    const Tensor &,
    const Tensor &,
    const std::vector<Tensor> &,
    const std::vector<Tensor> &,
    const std::vector<Tensor> &,
    size_t,
    size_t);

Tensor deepseek_moe(const Tensor &hidden,
                    const Tensor &topk_indices,
                    const Tensor &topk_weights,
                    const std::vector<Tensor> &gate_weights,
                    const std::vector<Tensor> &up_weights,
                    const std::vector<Tensor> &down_weights,
                    size_t intermediate_size,
                    size_t num_experts);

void deepseek_moe_(Tensor out,
                   const Tensor &hidden,
                   const Tensor &topk_indices,
                   const Tensor &topk_weights,
                   const std::vector<Tensor> &gate_weights,
                   const std::vector<Tensor> &up_weights,
                   const std::vector<Tensor> &down_weights,
                   size_t intermediate_size,
                   size_t num_experts);

} // namespace infinicore::op
