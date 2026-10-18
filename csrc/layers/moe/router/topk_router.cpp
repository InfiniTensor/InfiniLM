#include "../../../weights/weights.hpp"
#include "../../linear/dense.hpp"
#include "../../../execution/submit.hpp"
#include <base/topk_softmax.h>
#include "topk_router.hpp"

#include "runtime/runtime.hpp"

#include <exception>
#include <limits>
#include <optional>
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

TopKRouter::TopKRouter(infinilm::runtime::Runtime &runtime,
                       std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("TopKRouter construction requires eager Runtime");
    if (!model_config) throw std::invalid_argument("TopKRouter construction requires configuration");
    for (const char *key : {"num_experts", "num_experts_per_tok", "hidden_size"})
        if (model_config->get<int64_t>(key) <= 0)
            throw std::invalid_argument("TopKRouter construction requires positive dimensions");

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
    if (num_experts_per_tok_ > num_experts_)
        throw std::invalid_argument("TopKRouter requires num_experts_per_tok <= num_experts");
    if (router_backend_ == TopKRouterBackend::FusedGate) {
        if (!use_correction_bias_) throw std::runtime_error("fused_gate MoE router requires correction bias");
        if (num_expert_group_ == 0 || topk_group_ == 0)
            throw std::runtime_error("fused_gate MoE router requires num_expert_group/n_group and topk_group");
        if (num_experts_ % num_expert_group_ != 0)
            throw std::runtime_error("fused_gate MoE router requires num_experts divisible by num_expert_group");
        if (num_experts_per_tok_ <= num_fused_shared_experts_)
            throw std::runtime_error("fused_gate MoE router requires num_experts_per_tok > num_fused_shared_experts");
        }
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const auto dtype = model_config->get_dtype();
    const size_t bytes = infinilm::runtime::dsize(dtype), limit = std::numeric_limits<size_t>::max();
    if (!bytes || num_experts_ > limit / hidden_size || num_experts_ * hidden_size > limit / bytes
        || (use_correction_bias_ && num_experts_ > limit / infinilm::runtime::dsize(infinilm::runtime::DataType::kFloat32)))
        throw std::length_error("TopKRouter parameter size overflow");
    device_ = runtime.device();
    try {
        weight_ = infinilm::weights::Weight(runtime, {num_experts_, hidden_size}, dtype);
        this->bind_weight("weight", weight_);
        if (use_correction_bias_) {
            e_score_correction_bias_ = infinilm::weights::Weight(runtime, {num_experts_}, infinilm::runtime::DataType::kFloat32);
            this->bind_weight("e_score_correction_bias", e_score_correction_bias_);
        }
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

std::tuple<infinilm::runtime::Tensor, infinilm::runtime::Tensor> TopKRouter::forward(
    infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &hidden_states) const {
    ASSERT(hidden_states->ndim() == 2);

    size_t ntoken = hidden_states->shape()[0];
    auto router_logits = infinilm::layers::linear::dense_projection(runtime, hidden_states, weight_.tensor(), std::nullopt, 1.0f);

    auto router_scores = infinilm::runtime::Tensor::empty(runtime, {ntoken, num_experts_per_tok_}, infinilm::runtime::DataType::kFloat32);
    auto router_indices = infinilm::runtime::Tensor::empty(runtime, {ntoken, num_experts_per_tok_}, infinilm::runtime::DataType::kInt32);

    const infinilm::runtime::Tensor correction_bias = use_correction_bias_ ? static_cast<infinilm::runtime::Tensor>(e_score_correction_bias_) : infinilm::runtime::Tensor();

    switch (router_backend_) {
    case TopKRouterBackend::Softmax: {
        auto topk_scratch = infinilm::runtime::Tensor::empty(runtime, router_scores->shape(), infinilm::runtime::DataType::kInt32);
        infini::ops::TopkSoftmaxConfig topk_config{moe_softcapping_, 0.0f};
        topk_config.set_implementation_index(
            infini::ops::TopkSoftmax::DefaultConfig(runtime.device().type()).implementation_index());
        const auto topk_bias = correction_bias
            ? std::optional<infinilm::runtime::Tensor>{correction_bias}
            : std::optional<infinilm::runtime::Tensor>{};
        infinilm::execution::submit<infini::ops::TopkSoftmax>(
            runtime, infinilm::execution::RecordingPolicy::AllowHostReplay, topk_config,
            router_logits, topk_bias, std::optional<infinilm::runtime::Tensor>{},
            norm_topk_prob_, router_scores, router_indices, topk_scratch);
        INFINILM_TRACE_OP(MoeTopkSoftmax, router_scores);
        break;
    }
    case TopKRouterBackend::Sigmoid: {
        auto token_expert_indices = infinilm::runtime::Tensor::empty(
            runtime, router_scores->shape(), infinilm::runtime::DataType::kInt32);
        const auto sigmoid_bias = correction_bias
            ? std::optional<infinilm::runtime::Tensor>{correction_bias}
            : std::optional<infinilm::runtime::Tensor>{};
        infinilm::execution::submit<infini::ops::TopkSigmoid>(
            runtime, infinilm::execution::RecordingPolicy::AllowHostReplay,
            infini::ops::TopkSigmoid::DefaultConfig(runtime.device().type()),
            router_logits, sigmoid_bias, std::optional<infinilm::runtime::Tensor>{},
            norm_topk_prob_, 1.0, router_scores, router_indices, token_expert_indices);
        INFINILM_TRACE_OP(MoeTopkSigmoid, router_scores);
        break;
    }
    case TopKRouterBackend::FusedGate:
        infinilm::execution::submit<infini::ops::MoeFusedGate>(
            runtime,
            infinilm::execution::RecordingPolicy::AllowHostReplay,
            infini::ops::MoeFusedGate::DefaultConfig(router_scores->device().type()),
            router_scores,
            router_indices,
            router_logits,
            e_score_correction_bias_.tensor(),
            num_expert_group_,
            topk_group_,
            num_fused_shared_experts_,
            routed_scaling_factor_,
            apply_routed_scaling_factor_on_output_);
        INFINILM_TRACE_OP(MoeFusedGate, router_scores);
        break;
    }

    return std::make_tuple(router_scores, router_indices);
}

} // namespace infinilm::layers::moe
