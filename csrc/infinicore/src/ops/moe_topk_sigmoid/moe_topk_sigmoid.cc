#include "infinicore/ops/moe_topk_sigmoid.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(MoeTopkSigmoid);

MoeTopkSigmoid::MoeTopkSigmoid(Tensor topk_weights,
                               Tensor topk_indices,
                               const Tensor &gating_output,
                               const Tensor &correction_bias,
                               const bool renormalize) {
    if (correction_bias) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(topk_weights, topk_indices, gating_output, correction_bias);
    } else {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(topk_weights, topk_indices, gating_output);
    }
    INFINICORE_GRAPH_OP_DISPATCH(
        topk_weights->device().type(),
        topk_weights,
        topk_indices,
        gating_output,
        correction_bias,
        renormalize);
}

void MoeTopkSigmoid::execute(Tensor topk_weights,
                             Tensor topk_indices,
                             const Tensor &gating_output,
                             const Tensor &correction_bias,
                             const bool renormalize) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        MoeTopkSigmoid,
        topk_weights,
        topk_indices,
        gating_output,
        correction_bias,
        renormalize);
}

std::tuple<Tensor, Tensor> moe_topk_sigmoid(
    const Tensor &gating_output,
    size_t topk,
    bool renormalize,
    const Tensor &correction_bias) {
    auto shape = gating_output->shape();
    INFINICORE_ASSERT(shape.size() == 2);
    auto topk_weights = Tensor::empty({shape[0], topk}, DataType::kFloat32, gating_output->device());
    auto topk_indices = Tensor::empty({shape[0], topk}, DataType::kInt32, gating_output->device());
    moe_topk_sigmoid_(topk_weights, topk_indices, gating_output, correction_bias, renormalize);
    return {topk_weights, topk_indices};
}

void moe_topk_sigmoid_(Tensor topk_weights,
                       Tensor topk_indices,
                       const Tensor &gating_output,
                       const Tensor &correction_bias,
                       bool renormalize) {
    MoeTopkSigmoid::execute(topk_weights, topk_indices, gating_output, correction_bias, renormalize);
}

} // namespace infinicore::op
