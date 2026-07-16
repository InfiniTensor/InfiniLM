#include "infinicore/ops/moe_fused_gate.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(MoeFusedGate);

MoeFusedGate::MoeFusedGate(Tensor topk_weights,
                           Tensor topk_indices,
                           const Tensor &input,
                           const Tensor &bias,
                           const size_t num_expert_group,
                           const size_t topk_group,
                           const size_t num_fused_shared_experts,
                           const float routed_scaling_factor,
                           const bool apply_routed_scaling_factor_on_output) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(topk_weights, topk_indices, input, bias);
    INFINICORE_GRAPH_OP_DISPATCH(
        topk_weights->device().type(),
        topk_weights,
        topk_indices,
        input,
        bias,
        num_expert_group,
        topk_group,
        num_fused_shared_experts,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output);
}

void MoeFusedGate::execute(Tensor topk_weights,
                           Tensor topk_indices,
                           const Tensor &input,
                           const Tensor &bias,
                           const size_t num_expert_group,
                           const size_t topk_group,
                           const size_t num_fused_shared_experts,
                           const float routed_scaling_factor,
                           const bool apply_routed_scaling_factor_on_output) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        MoeFusedGate,
        topk_weights,
        topk_indices,
        input,
        bias,
        num_expert_group,
        topk_group,
        num_fused_shared_experts,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output);
}

std::tuple<Tensor, Tensor> moe_fused_gate(
    const Tensor &input,
    const Tensor &bias,
    size_t topk,
    size_t num_expert_group,
    size_t topk_group,
    size_t num_fused_shared_experts,
    float routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output) {
    auto shape = input->shape();
    INFINICORE_ASSERT(shape.size() == 2);
    auto topk_weights = Tensor::empty({shape[0], topk}, DataType::kFloat32, input->device());
    auto topk_indices = Tensor::empty({shape[0], topk}, DataType::kInt32, input->device());
    moe_fused_gate_(
        topk_weights,
        topk_indices,
        input,
        bias,
        num_expert_group,
        topk_group,
        num_fused_shared_experts,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output);
    return {topk_weights, topk_indices};
}

void moe_fused_gate_(Tensor topk_weights,
                     Tensor topk_indices,
                     const Tensor &input,
                     const Tensor &bias,
                     size_t num_expert_group,
                     size_t topk_group,
                     size_t num_fused_shared_experts,
                     float routed_scaling_factor,
                     bool apply_routed_scaling_factor_on_output) {
    MoeFusedGate::execute(
        topk_weights,
        topk_indices,
        input,
        bias,
        num_expert_group,
        topk_group,
        num_fused_shared_experts,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output);
}

} // namespace infinicore::op
