#include "infinicore/ops/moe_fused_dense.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(MoeFusedDense);

MoeFusedDense::MoeFusedDense(Tensor output,
                             const Tensor &hidden_states,
                             const Tensor &w13,
                             const Tensor &w2,
                             const Tensor &topk_weights,
                             const Tensor &topk_ids,
                             const Tensor &sorted_token_ids,
                             const Tensor &expert_ids,
                             const Tensor &num_tokens_post_padded) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(
        output, hidden_states, w13, w2, topk_weights, topk_ids,
        sorted_token_ids, expert_ids, num_tokens_post_padded);
    INFINICORE_GRAPH_OP_DISPATCH(
        output->device().type(), output, hidden_states, w13, w2, topk_weights, topk_ids,
        sorted_token_ids, expert_ids, num_tokens_post_padded);
}

void MoeFusedDense::execute(Tensor output,
                            const Tensor &hidden_states,
                            const Tensor &w13,
                            const Tensor &w2,
                            const Tensor &topk_weights,
                            const Tensor &topk_ids,
                            const Tensor &sorted_token_ids,
                            const Tensor &expert_ids,
                            const Tensor &num_tokens_post_padded) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        MoeFusedDense, output, hidden_states, w13, w2, topk_weights, topk_ids,
        sorted_token_ids, expert_ids, num_tokens_post_padded);
}

Tensor moe_fused_dense(
    const Tensor &hidden_states,
    const Tensor &w13,
    const Tensor &w2,
    const Tensor &topk_weights,
    const Tensor &topk_ids,
    const Tensor &sorted_token_ids,
    const Tensor &expert_ids,
    const Tensor &num_tokens_post_padded) {
    auto shape = hidden_states->shape();
    INFINICORE_ASSERT(shape.size() == 2);
    auto output = Tensor::empty(shape, hidden_states->dtype(), hidden_states->device());
    moe_fused_dense_(
        output, hidden_states, w13, w2, topk_weights, topk_ids,
        sorted_token_ids, expert_ids, num_tokens_post_padded);
    return output;
}

void moe_fused_dense_(
    Tensor output,
    const Tensor &hidden_states,
    const Tensor &w13,
    const Tensor &w2,
    const Tensor &topk_weights,
    const Tensor &topk_ids,
    const Tensor &sorted_token_ids,
    const Tensor &expert_ids,
    const Tensor &num_tokens_post_padded) {
    MoeFusedDense::execute(
        output, hidden_states, w13, w2, topk_weights, topk_ids,
        sorted_token_ids, expert_ids, num_tokens_post_padded);
}

} // namespace infinicore::op
