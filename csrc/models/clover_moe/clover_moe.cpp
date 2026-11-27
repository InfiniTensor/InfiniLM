#include "clover_moe.hpp"
#include "../../global_state/forward_context.hpp"
#include "../../layers/linear/dense.hpp"
#include "execution/submit.hpp"
#include <base/clover_coreset.h>
#include <base/clover_mask_routes.h>
#include <base/clover_merge_moe.h>

namespace infinilm::models::clover_moe {

CloverSparseMoe::CloverSparseMoe(runtime::Runtime &runtime, std::shared_ptr<config::ModelConfig> config,
                                 size_t layer, const engine::distributed::RankInfo &rank,
                                 backends::AttentionBackend backend, const layers::moe::EPConfig &ep)
    : Qwen3NextSparseMoeBlock(runtime, config, layer, rank, backend, ep),
      coreset_(config->get_or<bool>("moe_block_coreset_enabled", true)),
      top_p_(config->get_or<double>("moe_block_coreset_top_p", 0.9)),
      top_k_(config->get<size_t>("num_experts_per_tok")) {
    router_weight_ = gate_->state_dict().at("weight").tensor();
}

std::tuple<runtime::Tensor, runtime::Tensor> CloverSparseMoe::route(
    runtime::Runtime &runtime, const runtime::Tensor &hidden,
    const global_state::ForwardContext &context) const {
    if (!context.diffusion_preview || (!coreset_ && !context.diffusion_expert_budget)) {
        return gate_->forward(runtime, hidden);
    }
    auto logits = layers::linear::dense_projection(runtime, hidden, router_weight_, std::nullopt, 1.0f);
    auto masked = logits;
    const auto device = runtime.device().type();
    if (coreset_) {
        masked = runtime::Tensor::empty(runtime, logits->shape(), logits->dtype());
        execution::submit<infini::ops::CloverCoreset>(runtime, execution::RecordingPolicy::HostOnly,
                                                      infini::ops::CloverCoreset::DefaultConfig(device), logits, context.diffusion_block_size, top_p_, masked);
    }
    const auto active_top_k = context.diffusion_expert_budget ? size_t{4} : top_k_;
    auto weights = runtime::Tensor::empty(runtime, {hidden->size(0), active_top_k}, runtime::DataType::kFloat32);
    auto indices = runtime::Tensor::empty(runtime, weights->shape(), runtime::DataType::kInt32);
    auto scratch = runtime::Tensor::empty(runtime, weights->shape(), runtime::DataType::kInt32);
    execution::submit<infini::ops::TopkSoftmax>(runtime, execution::RecordingPolicy::HostOnly,
                                                infini::ops::TopkSoftmax::DefaultConfig(device), masked,
                                                std::optional<runtime::Tensor>{}, std::optional<runtime::Tensor>{}, true,
                                                weights, indices, scratch);
    return {weights, indices};
}

infinilm::runtime::Tensor CloverSparseMoe::forward(infinilm::runtime::Runtime &runtime,
                                                   const infinilm::runtime::Tensor &hidden_states,
                                                   const global_state::ForwardContext &context) const {
    ASSERT(hidden_states->ndim() == 3);

    auto shape = hidden_states->shape();
    auto hidden_states_reshaped = hidden_states->view({shape[0] * shape[1], shape[2]});

    auto [routing_weights, selected_experts] = route(runtime, hidden_states_reshaped, context);
    if (context.diffusion_preview && context.diffusion_expert_budget) {
        execution::submit<infini::ops::CloverMaskRoutes>(runtime, execution::RecordingPolicy::HostOnly,
                                                         infini::ops::CloverMaskRoutes::DefaultConfig(runtime.device().type()),
                                                         selected_experts, context.diffusion_live, int64_t{256}, selected_experts);
    }
    infinilm::layers::moe::TopKOutput topk_output{
        routing_weights,
        selected_experts,
        infinilm::runtime::Tensor(),
    };
    auto routed_states = fused_moe_->forward(
        runtime, hidden_states_reshaped,
        topk_output,
        experts_->moe_weights());
    if (context.diffusion_preview && context.diffusion_expert_budget) {
        if (!cached_routed_ || cached_routed_->shape() != routed_states->shape()) {
            cached_routed_ = runtime::Tensor::empty(runtime, routed_states->shape(), routed_states->dtype());
        }
        execution::submit<infini::ops::CloverMergeMoe>(runtime, execution::RecordingPolicy::HostOnly,
                                                       infini::ops::CloverMergeMoe::DefaultConfig(runtime.device().type()),
                                                       routed_states, context.diffusion_live, cached_routed_, routed_states);
    }

    auto shared_states = shared_expert_->forward(runtime, hidden_states);
    auto hidden_states_for_gate = hidden_states;
    const auto ops_input = shared_expert_gate_->forward(runtime, hidden_states_for_gate);
    auto shared_gate = infinilm::runtime::Tensor::empty(runtime, ops_input->shape(), ops_input->dtype());
    infinilm::execution::submit<infini::ops::Sigmoid>(
        runtime, infini::ops::Sigmoid::DefaultConfig(runtime.device().type()), ops_input, shared_gate);
    INFINILM_TRACE_OP(Sigmoid, shared_gate);
    shared_gate = shared_gate->as_strided(shared_states->shape(), {shared_gate->stride(0), shared_gate->stride(1), 0});
    {
        auto ops_output = infinilm::runtime::Tensor::empty(runtime, shared_states->shape(), shared_states->dtype());
        infinilm::execution::submit<infini::ops::Mul>(
            runtime, infini::ops::Mul::DefaultConfig(runtime.device().type()), shared_states, shared_gate, ops_output);
        INFINILM_TRACE_OP(Mul, ops_output);
        shared_states = std::move(ops_output);
    }

    auto routed_states_3d = routed_states->as_strided(
        {shape[0], shape[1], shape[2]},
        {static_cast<infinilm::runtime::Stride>(shape[1] * shape[2]), static_cast<infinilm::runtime::Stride>(shape[2]), 1});
    auto ops_output = infinilm::runtime::Tensor::empty(runtime, routed_states_3d->shape(), routed_states_3d->dtype());
    infinilm::execution::submit<infini::ops::Add>(
        runtime, infini::ops::Add::DefaultConfig(runtime.device().type()), routed_states_3d, shared_states, ops_output);
    INFINILM_TRACE_OP(Add, ops_output);
    return ops_output;
}

} // namespace infinilm::models::clover_moe
