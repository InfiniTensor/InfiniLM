#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

namespace infinicore::op {

// Device-graph safety is ctor-gated (moeTritonCaptureAllowed + QY policy).
// Default: MoE-in-graph decode FULL (vLLM align). Kill-switch
// INFINI_MOE_QY_ALLOW_GRAPH=0 forces Decode host-break. MoE also prefers its
// own device segment to avoid Denglin CUresult 701 in giant decode captures.
class MoeFusedDense : public graph::DispatchableGraphOperator {
public:
    using schema = void (*)(Tensor,
                            const Tensor &,
                            const Tensor &,
                            const Tensor &,
                            const Tensor &,
                            const Tensor &,
                            const Tensor &,
                            const Tensor &,
                            const Tensor &);
    using plan_schema = void *(*)(Tensor,
                                  const Tensor &,
                                  const Tensor &,
                                  const Tensor &,
                                  const Tensor &,
                                  const Tensor &,
                                  const Tensor &,
                                  const Tensor &,
                                  const Tensor &);

    static common::OpDispatcher<plan_schema> &plan_dispatcher();
    static common::OpDispatcher<run_schema> &run_dispatcher();
    static common::OpDispatcher<cleanup_schema> &cleanup_dispatcher();

    MoeFusedDense(Tensor output,
                  const Tensor &hidden_states,
                  const Tensor &w13,
                  const Tensor &w2,
                  const Tensor &topk_weights,
                  const Tensor &topk_ids,
                  const Tensor &sorted_token_ids,
                  const Tensor &expert_ids,
                  const Tensor &num_tokens_post_padded);

    static void execute(Tensor output,
                        const Tensor &hidden_states,
                        const Tensor &w13,
                        const Tensor &w2,
                        const Tensor &topk_weights,
                        const Tensor &topk_ids,
                        const Tensor &sorted_token_ids,
                        const Tensor &expert_ids,
                        const Tensor &num_tokens_post_padded);

    bool is_device_graph_capture_safe() const override {
        return device_graph_capture_safe_;
    }

    // vLLM CUDAGraphMode.FULL records MoE inside one decode graph — do not
    // isolate MoE into private InfiniRT segments (that left wall at ~5.4 tok/s).
    bool prefers_own_device_segment() const override { return false; }

private:
    bool device_graph_capture_safe_;
};

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
