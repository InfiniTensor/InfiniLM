#include "infinicore/ops/moe_topk_softmax.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(MoeTopkSoftmax);

MoeTopkSoftmax::MoeTopkSoftmax(Tensor topk_weights,
                               Tensor topk_indices,
                               const Tensor &gating_output,
                               const Tensor &correction_bias,
                               const bool renormalize,
                               const float moe_softcapping) {
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
        renormalize,
        moe_softcapping);
}

void MoeTopkSoftmax::execute(Tensor topk_weights,
                             Tensor topk_indices,
                             const Tensor &gating_output,
                             const Tensor &correction_bias,
                             const bool renormalize,
                             const float moe_softcapping) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        MoeTopkSoftmax,
        topk_weights,
        topk_indices,
        gating_output,
        correction_bias,
        renormalize,
        moe_softcapping);
}

std::tuple<Tensor, Tensor> moe_topk_softmax(
    const Tensor &gating_output,
    size_t topk,
    bool renormalize,
    float moe_softcapping,
    const Tensor &correction_bias) {
    auto shape = gating_output->shape();
    INFINICORE_ASSERT(shape.size() == 2);
    auto topk_weights = Tensor::empty({shape[0], topk}, DataType::kFloat32, gating_output->device());
    auto topk_indices = Tensor::empty({shape[0], topk}, DataType::kInt32, gating_output->device());
    moe_topk_softmax_(topk_weights, topk_indices, gating_output, correction_bias, renormalize, moe_softcapping);
    return {topk_weights, topk_indices};
}

void moe_topk_softmax_(Tensor topk_weights,
                       Tensor topk_indices,
                       const Tensor &gating_output,
                       const Tensor &correction_bias,
                       bool renormalize,
                       float moe_softcapping) {
    MoeTopkSoftmax::execute(topk_weights, topk_indices, gating_output, correction_bias, renormalize, moe_softcapping);
}

} // namespace infinicore::op
