#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

#include <tuple>

namespace infinicore::op {

// Same QY device-graph gate as MoeFusedDense (default MoE-in-graph; =0 kill-switch).
class MoeFusedGate : public graph::DispatchableGraphOperator {
public:
    using schema = void (*)(Tensor,
                            Tensor,
                            const Tensor &,
                            const Tensor &,
                            const size_t,
                            const size_t,
                            const size_t,
                            const float,
                            const bool);
    using plan_schema = void *(*)(Tensor,
                                  Tensor,
                                  const Tensor &,
                                  const Tensor &,
                                  const size_t,
                                  const size_t,
                                  const size_t,
                                  const float,
                                  const bool);

    static common::OpDispatcher<plan_schema> &plan_dispatcher();
    static common::OpDispatcher<run_schema> &run_dispatcher();
    static common::OpDispatcher<cleanup_schema> &cleanup_dispatcher();

    MoeFusedGate(Tensor topk_weights,
                 Tensor topk_indices,
                 const Tensor &input,
                 const Tensor &bias,
                 const size_t num_expert_group,
                 const size_t topk_group,
                 const size_t num_fused_shared_experts,
                 const float routed_scaling_factor,
                 const bool apply_routed_scaling_factor_on_output);

    static void execute(Tensor topk_weights,
                        Tensor topk_indices,
                        const Tensor &input,
                        const Tensor &bias,
                        const size_t num_expert_group,
                        const size_t topk_group,
                        const size_t num_fused_shared_experts,
                        const float routed_scaling_factor,
                        const bool apply_routed_scaling_factor_on_output);

    bool is_device_graph_capture_safe() const override {
        return device_graph_capture_safe_;
    }

    // Match MoeFusedDense: stay in the monolithic decode FULL capture.
    bool prefers_own_device_segment() const override { return false; }

private:
    bool device_graph_capture_safe_;
};

std::tuple<Tensor, Tensor> moe_fused_gate(
    const Tensor &input,
    const Tensor &bias,
    size_t topk,
    size_t num_expert_group,
    size_t topk_group,
    size_t num_fused_shared_experts,
    float routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output);

void moe_fused_gate_(
    Tensor topk_weights,
    Tensor topk_indices,
    const Tensor &input,
    const Tensor &bias,
    size_t num_expert_group,
    size_t topk_group,
    size_t num_fused_shared_experts,
    float routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output);

} // namespace infinicore::op
