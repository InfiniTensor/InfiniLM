#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

namespace infinicore::op {

// QY MoE runs through ATen + _dl_C and is not CUDA-graph capture safe.
// Keep it in eager graph segments so FA/gemm can still capture around it.
INFINICORE_GRAPH_OP_CLASS_CAPTURE_UNSAFE(MoeFusedDense,
                                         Tensor,
                                         const Tensor &,
                                         const Tensor &,
                                         const Tensor &,
                                         const Tensor &,
                                         const Tensor &,
                                         const Tensor &,
                                         const Tensor &,
                                         const Tensor &);

Tensor moe_fused_dense(
    const Tensor &hidden_states,
    const Tensor &w13,
    const Tensor &w2,
    const Tensor &topk_weights,
    const Tensor &topk_ids,
    const Tensor &sorted_token_ids,
    const Tensor &expert_ids,
    const Tensor &num_tokens_post_padded);

void moe_fused_dense_(
    Tensor output,
    const Tensor &hidden_states,
    const Tensor &w13,
    const Tensor &w2,
    const Tensor &topk_weights,
    const Tensor &topk_ids,
    const Tensor &sorted_token_ids,
    const Tensor &expert_ids,
    const Tensor &num_tokens_post_padded);

} // namespace infinicore::op
