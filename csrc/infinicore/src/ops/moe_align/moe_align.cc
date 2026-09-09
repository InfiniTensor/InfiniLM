#include "infinicore/ops/moe_align.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(MoeAlign);
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(MoeAlignWithExpertMap);

MoeAlign::MoeAlign(Tensor sorted_token_ids,
                   Tensor expert_ids,
                   Tensor num_tokens_post_padded,
                   const Tensor &topk_ids,
                   const size_t num_experts,
                   const size_t block_size,
                   const bool pad_sorted_token_ids) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(sorted_token_ids, expert_ids, num_tokens_post_padded, topk_ids);
    INFINICORE_GRAPH_OP_DISPATCH(
        sorted_token_ids->device().type(),
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_ids,
        num_experts,
        block_size,
        pad_sorted_token_ids);
}

void MoeAlign::execute(Tensor sorted_token_ids,
                       Tensor expert_ids,
                       Tensor num_tokens_post_padded,
                       const Tensor &topk_ids,
                       const size_t num_experts,
                       const size_t block_size,
                       const bool pad_sorted_token_ids) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        MoeAlign,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_ids,
        num_experts,
        block_size,
        pad_sorted_token_ids);
}

MoeAlignWithExpertMap::MoeAlignWithExpertMap(Tensor sorted_token_ids,
                                             Tensor expert_ids,
                                             Tensor num_tokens_post_padded,
                                             const Tensor &topk_ids,
                                             const Tensor &expert_map,
                                             const size_t num_experts,
                                             const size_t block_size,
                                             const bool pad_sorted_token_ids) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(sorted_token_ids, expert_ids, num_tokens_post_padded, topk_ids, expert_map);
    INFINICORE_GRAPH_OP_DISPATCH(
        sorted_token_ids->device().type(),
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_ids,
        expert_map,
        num_experts,
        block_size,
        pad_sorted_token_ids);
}

void MoeAlignWithExpertMap::execute(Tensor sorted_token_ids,
                                    Tensor expert_ids,
                                    Tensor num_tokens_post_padded,
                                    const Tensor &topk_ids,
                                    const Tensor &expert_map,
                                    const size_t num_experts,
                                    const size_t block_size,
                                    const bool pad_sorted_token_ids) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        MoeAlignWithExpertMap,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_ids,
        expert_map,
        num_experts,
        block_size,
        pad_sorted_token_ids);
}

std::tuple<Tensor, Tensor, Tensor> moe_align(
    const Tensor &topk_ids,
    size_t num_experts,
    size_t block_size,
    bool pad_sorted_token_ids) {
    auto shape = topk_ids->shape();
    INFINICORE_ASSERT(shape.size() == 2);
    const size_t numel = shape[0] * shape[1];
    const size_t align_num_experts = num_experts + 1;
    const size_t max_num_tokens_padded = numel < align_num_experts
                                           ? numel * block_size
                                           : numel + align_num_experts * (block_size - 1);
    const size_t sorted_token_ids_capacity = ((max_num_tokens_padded + 3) / 4) * 4;
    const size_t max_num_blocks = (max_num_tokens_padded + block_size - 1) / block_size;

    auto sorted_token_ids = Tensor::empty({sorted_token_ids_capacity}, DataType::kInt32, topk_ids->device());
    auto expert_ids = Tensor::empty({max_num_blocks}, DataType::kInt32, topk_ids->device());
    auto num_tokens_post_padded = Tensor::empty({1}, DataType::kInt32, topk_ids->device());

    moe_align_(
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_ids,
        num_experts,
        block_size,
        pad_sorted_token_ids);

    return {sorted_token_ids, expert_ids, num_tokens_post_padded};
}

void moe_align_(Tensor sorted_token_ids,
                Tensor expert_ids,
                Tensor num_tokens_post_padded,
                const Tensor &topk_ids,
                size_t num_experts,
                size_t block_size,
                bool pad_sorted_token_ids) {
    MoeAlign::execute(
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_ids,
        num_experts,
        block_size,
        pad_sorted_token_ids);
}

void moe_align_with_expert_map_(Tensor sorted_token_ids,
                                Tensor expert_ids,
                                Tensor num_tokens_post_padded,
                                const Tensor &topk_ids,
                                const Tensor &expert_map,
                                size_t num_experts,
                                size_t block_size,
                                bool pad_sorted_token_ids) {
    MoeAlignWithExpertMap::execute(
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_ids,
        expert_map,
        num_experts,
        block_size,
        pad_sorted_token_ids);
}

} // namespace infinicore::op
