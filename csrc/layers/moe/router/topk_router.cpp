#include "topk_router.hpp"

#include "infinicore/ops.hpp"

#include <stdexcept>
#include <string>

namespace infinilm::layers::moe {
namespace {

TopKRouterBackend parse_router_backend(const std::string &backend) {
    if (backend == "softmax") {
        return TopKRouterBackend::Softmax;
    }
    if (backend == "sigmoid") {
        return TopKRouterBackend::Sigmoid;
    }
    if (backend == "fused_gate" || backend == "noaux_tc") {
        return TopKRouterBackend::FusedGate;
    }
    throw std::runtime_error("Unsupported MoE router backend: " + backend);
}

std::string router_backend_name(const std::shared_ptr<infinilm::config::ModelConfig> &model_config) {
    auto backend = model_config->get_or<std::string>({"moe_router_backend"}, "");
    if (!backend.empty()) {
        return backend;
    }
    if (model_config->get_or<std::string>("topk_method", "") == "noaux_tc") {
        return "fused_gate";
    }
    backend = model_config->get_or<std::string>({"scoring_func"}, "");
    if (!backend.empty()) {
        return backend;
    }
    return "softmax";
}

} // namespace

TopKRouter::TopKRouter(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                       const infinicore::Device &device) {
    num_experts_ = model_config->get<size_t>("num_experts");
    num_experts_per_tok_ = model_config->get<size_t>("num_experts_per_tok");
    norm_topk_prob_ = model_config->get_or<bool>("norm_topk_prob", false);
    routed_scaling_factor_ = model_config->get_or<float>("routed_scaling_factor", 1.0f);
    moe_softcapping_ = model_config->get_or<float>("moe_softcapping", 0.0f);
    num_expert_group_ = model_config->get_or_alias<size_t>("num_expert_group", "n_group", 0);
    topk_group_ = model_config->get_or<size_t>("topk_group", 0);
    num_fused_shared_experts_ = model_config->get_or<size_t>("num_fused_shared_experts", 0);
    apply_routed_scaling_factor_on_output_ = model_config->get_or<bool>("apply_routed_scaling_factor_on_output", false);
    router_backend_ = parse_router_backend(router_backend_name(model_config));
    use_correction_bias_ = model_config->get_or<bool>({"e_score_correction_bias", "moe_router_use_correction_bias"}, false) || router_backend_ == TopKRouterBackend::FusedGate;
    ASSERT((num_experts_ > 0) && (num_experts_per_tok_ > 0) && (num_experts_per_tok_ <= num_experts_));

    INFINICORE_NN_PARAMETER_INIT(
        weight,
        ({num_experts_, model_config->get<size_t>("hidden_size")}, model_config->get_dtype(), device));

    if (use_correction_bias_) {
        INFINICORE_NN_PARAMETER_INIT(e_score_correction_bias, ({num_experts_}, infinicore::DataType::F32, device));
    }

    if (router_backend_ == TopKRouterBackend::FusedGate) {
        if (!use_correction_bias_) {
            throw std::runtime_error("fused_gate MoE router requires correction bias");
        }
        if (num_expert_group_ == 0 || topk_group_ == 0) {
            throw std::runtime_error("fused_gate MoE router requires num_expert_group/n_group and topk_group");
        }
        if (num_experts_ % num_expert_group_ != 0) {
            throw std::runtime_error("fused_gate MoE router requires num_experts divisible by num_expert_group");
        }
        if (num_experts_per_tok_ <= num_fused_shared_experts_) {
            throw std::runtime_error("fused_gate MoE router requires num_experts_per_tok > num_fused_shared_experts");
        }
    }
}

std::tuple<infinicore::Tensor, infinicore::Tensor> TopKRouter::forward(const infinicore::Tensor &hidden_states) const {
    ASSERT(hidden_states->ndim() == 2);

    size_t ntoken = hidden_states->shape()[0];
    auto router_logits = infinicore::op::linear(hidden_states, weight_, std::nullopt, 1.0f);

    auto router_scores = infinicore::Tensor::empty({ntoken, num_experts_per_tok_}, infinicore::DataType::F32, hidden_states->device());
    auto router_indices = infinicore::Tensor::empty({ntoken, num_experts_per_tok_}, infinicore::DataType::I32, hidden_states->device());

    const infinicore::Tensor correction_bias = use_correction_bias_ ? static_cast<infinicore::Tensor>(e_score_correction_bias_) : infinicore::Tensor();

    switch (router_backend_) {
    case TopKRouterBackend::Softmax:
        infinicore::op::moe_topk_softmax_(
            router_scores,
            router_indices,
            router_logits,
            correction_bias,
            norm_topk_prob_,
            moe_softcapping_);
        break;
    case TopKRouterBackend::Sigmoid:
        infinicore::op::moe_topk_sigmoid_(
            router_scores,
            router_indices,
            router_logits,
            correction_bias,
            norm_topk_prob_);
        break;
    case TopKRouterBackend::FusedGate:
        infinicore::op::moe_fused_gate_(
            router_scores,
            router_indices,
            router_logits,
            e_score_correction_bias_,
            num_expert_group_,
            topk_group_,
            num_fused_shared_experts_,
            routed_scaling_factor_,
            apply_routed_scaling_factor_on_output_);
        break;
    }

    return std::make_tuple(router_scores, router_indices);
}

} // namespace infinilm::layers::moe
