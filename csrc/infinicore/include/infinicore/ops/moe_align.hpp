#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

#include <tuple>

namespace infinicore::op {

// QY uses the ATen host-path implementation (CPU round-trips); not capture-safe.
INFINICORE_GRAPH_OP_CLASS_CAPTURE_UNSAFE(MoeAlign, Tensor, Tensor, Tensor, const Tensor &, const size_t, const size_t, const bool);
INFINICORE_GRAPH_OP_CLASS_CAPTURE_UNSAFE(MoeAlignWithExpertMap, Tensor, Tensor, Tensor, const Tensor &, const Tensor &, const size_t, const size_t, const bool);

std::tuple<Tensor, Tensor, Tensor> moe_align(
    const Tensor &topk_ids,
    size_t num_experts,
    size_t block_size,
    bool pad_sorted_token_ids = true);

void moe_align_(
    Tensor sorted_token_ids,
    Tensor expert_ids,
    Tensor num_tokens_post_padded,
    const Tensor &topk_ids,
    size_t num_experts,
    size_t block_size,
    bool pad_sorted_token_ids = true);

void moe_align_with_expert_map_(
    Tensor sorted_token_ids,
    Tensor expert_ids,
    Tensor num_tokens_post_padded,
    const Tensor &topk_ids,
    const Tensor &expert_map,
    size_t num_experts,
    size_t block_size,
    bool pad_sorted_token_ids = true);

} // namespace infinicore::op
