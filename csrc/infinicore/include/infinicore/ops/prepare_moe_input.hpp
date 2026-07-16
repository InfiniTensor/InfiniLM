#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(PrepareMoeInput,
                          Tensor,
                          Tensor,
                          Tensor,
                          Tensor,
                          Tensor,
                          Tensor,
                          const Tensor &,
                          const size_t,
                          const size_t,
                          const size_t);

struct PrepareMoeInputOutput {
    Tensor expert_offsets;
    Tensor problem_sizes1;
    Tensor problem_sizes2;
    Tensor input_permutation;
    Tensor output_permutation;
};

PrepareMoeInputOutput prepare_moe_input(
    const Tensor &topk_ids,
    size_t num_experts,
    size_t n,
    size_t k);

void prepare_moe_input_(
    Tensor expert_offsets,
    Tensor blockscale_offsets,
    Tensor problem_sizes1,
    Tensor problem_sizes2,
    Tensor input_permutation,
    Tensor output_permutation,
    const Tensor &topk_ids,
    size_t num_experts,
    size_t n,
    size_t k);

} // namespace infinicore::op
