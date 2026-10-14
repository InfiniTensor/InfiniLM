#include "minicpm5_moe_sparse_moe_block.hpp"

#include <infinicore/ops/add.hpp>
#include <infinicore/ops/mul_scalar.hpp>

namespace infinilm::models::minicpm5_moe {

std::shared_ptr<infinilm::config::ModelConfig>
make_minicpm5_mlp_config(const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
                         size_t intermediate_size) {
    auto config_json = model_config->get_config_json();
    config_json["intermediate_size"] = intermediate_size;
    return std::make_shared<infinilm::config::ModelConfig>(std::move(config_json));
}

MiniCPM5MoESparseMoeBlock::MiniCPM5MoESparseMoeBlock(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    size_t layer_idx,
    const infinicore::Device &device) {
    INFINICORE_NN_MODULE_INIT(gate, model_config, device);
    INFINICORE_NN_MODULE_INIT(experts, model_config, device);
    INFINICORE_NN_MODULE_INIT(fused_moe, model_config, device, layer_idx);

    routed_scaling_factor_ = model_config->get_or<float>("routed_scaling_factor", 1.0f);
    const size_t n_shared_experts = model_config->get_or<size_t>("n_shared_experts", 0);
    has_shared_experts_ = n_shared_experts > 0;
    if (has_shared_experts_) {
        const size_t shared_intermediate_size = model_config->get<size_t>("moe_intermediate_size") * n_shared_experts;
        auto shared_config = make_minicpm5_mlp_config(model_config, shared_intermediate_size);
        INFINICORE_NN_MODULE_INIT(shared_experts, shared_config, device);
    }
}

infinicore::Tensor MiniCPM5MoESparseMoeBlock::forward(
    const infinicore::Tensor &hidden_states) const {
    ASSERT(hidden_states->ndim() == 3);
    const auto shape = hidden_states->shape();
    auto flattened = hidden_states->view({shape[0] * shape[1], shape[2]});

    auto [routing_weights, selected_experts] = gate_->forward(flattened);
    // Fold routed_scaling into topk weights (M×topk) instead of mul_scalar on
    // the full hidden [B,S,H] — Denglin ATen mul_Scalar OORs on long prefill.
    if (routed_scaling_factor_ != 1.0f) {
        routing_weights = infinicore::op::mul_scalar(routing_weights, routed_scaling_factor_);
    }
    infinilm::layers::moe::TopKOutput topk_output{
        routing_weights,
        selected_experts,
        infinicore::Tensor(),
    };
    auto routed = fused_moe_->forward(flattened, topk_output, experts_->moe_weights())->view(shape);
    if (has_shared_experts_) {
        return infinicore::op::add(routed, shared_experts_->forward(hidden_states));
    }
    return routed;
}

} // namespace infinilm::models::minicpm5_moe
