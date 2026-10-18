#include "execution/submit.hpp"
#include "kimi_k25_moe.hpp"
#include <exception>
#include "runtime/runtime.hpp"

#include "../../global_state/global_state.hpp"


#include <stdexcept>
#include <string>

namespace infinilm::models::kimi_k25 {
namespace {

bool uses_mxfp4(const std::shared_ptr<infinilm::config::ModelConfig> &model_config) {
    const auto &config = model_config->get_config_json();
    return config.contains("quantization_config")
        && config.at("quantization_config").is_object()
        && config.at("quantization_config").value("quant_method", "") == "quark";
}

} // namespace

KimiK25MXFP4Experts::KimiK25MXFP4Experts(
    infinilm::runtime::Runtime &runtime,
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const engine::distributed::RankInfo &rank_info) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("KimiK25MXFP4Experts construction requires eager Runtime");
    if (!model_config || rank_info.device != runtime.device() || rank_info.tp_size <= 0
        || rank_info.tp_rank < 0 || rank_info.tp_rank >= rank_info.tp_size
        || (rank_info.tp_size > 1 && !rank_info.comm))
        throw std::invalid_argument("Invalid explicit KimiK25MXFP4Experts construction resources");
    try {
        num_experts_ = model_config->get<size_t>("num_experts");
        hidden_size_ = model_config->get<size_t>("hidden_size");
        dtype_ = model_config->get_dtype();
        device_ = runtime.device();
    const size_t intermediate_size = model_config->get<size_t>("moe_intermediate_size");
    const size_t tp_size = static_cast<size_t>(rank_info.tp_size);
    const size_t tp_rank = static_cast<size_t>(rank_info.tp_rank);
    if (hidden_size_ % 32 != 0 || intermediate_size % (tp_size * 32) != 0) {
        throw std::runtime_error(
            "KimiK25MXFP4Experts: expert dimensions must be divisible by TP and MXFP4 group size");
    }
    local_intermediate_size_ = intermediate_size / tp_size;
    packed_parameters_.reserve(num_experts_ * 6);

    auto register_packed = [&](const std::string &name,
                               const infinilm::runtime::Shape &shape,
                               size_t tp_dim) {
        packed_parameters_.emplace_back(
                runtime, shape, infinilm::runtime::DataType::kUInt8, tp_dim, tp_rank, tp_size);
            this->bind_weight(name, packed_parameters_.back());
    };

    for (size_t expert_idx = 0; expert_idx < num_experts_; ++expert_idx) {
        const std::string prefix = std::to_string(expert_idx) + ".";
        for (const std::string projection : {"gate_proj", "up_proj"}) {
            register_packed(prefix + projection + ".weight",
                            {intermediate_size, hidden_size_ / 2}, 0);
            register_packed(prefix + projection + ".weight_scale",
                            {intermediate_size, hidden_size_ / 32}, 0);
        }
        register_packed(prefix + "down_proj.weight",
                        {hidden_size_, intermediate_size / 2}, 1);
        register_packed(prefix + "down_proj.weight_scale",
                        {hidden_size_, intermediate_size / 32}, 1);
    }
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

const infinilm::layers::moe::MoeWeights &KimiK25MXFP4Experts::moe_weights() const {
    if (!moe_weights_.packed_w13 || !moe_weights_.packed_w2) {
        throw std::runtime_error(
            "KimiK25MXFP4Experts: weights have not been processed after loading");
    }
    return moe_weights_;
}

void KimiK25MXFP4Experts::process_weights_after_loading(infinilm::runtime::Runtime &, bool) {
    // No real explicit-Runtime MXFP4 dequantization provider is available.
    // Reject before allocation/submission or clearing/publishing any weights.
                throw std::runtime_error(
        "KimiK25MXFP4Experts: typed post-load is unsupported without a real "
        "explicit-Runtime MXFP4 dequantization provider; packed weights/scales are retained");
}

std::shared_ptr<infinilm::config::ModelConfig>
make_kimi_mlp_config(
    const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
    size_t intermediate_size) {
    auto config_json = model_config->get_config_json();
    config_json["intermediate_size"] = intermediate_size;
    return std::make_shared<infinilm::config::ModelConfig>(std::move(config_json));
}

KimiK25MoE::KimiK25MoE(
    infinilm::runtime::Runtime &runtime,
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
                       size_t layer_idx,
    const engine::distributed::RankInfo &rank_info,
    backends::AttentionBackend attention_backend,
    const layers::moe::EPConfig &ep_config) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("KimiK25MoE construction requires eager Runtime");
    if (!model_config || rank_info.device != runtime.device() || rank_info.tp_size <= 0
        || rank_info.tp_rank < 0 || rank_info.tp_rank >= rank_info.tp_size
        || (rank_info.tp_size > 1 && !rank_info.comm))
        throw std::invalid_argument("Invalid explicit KimiK25MoE construction resources");
    try {
        gate_ = std::make_shared<std::remove_reference_t<decltype(*gate_)>>(
            runtime, model_config);
        this->bind_child("gate", gate_);
    routed_scaling_factor_ = model_config->get_or<float>("routed_scaling_factor", 1.0f);
    uses_mxfp4_experts_ = uses_mxfp4(model_config);
    if (uses_mxfp4_experts_) {
            mxfp4_experts_ = std::make_shared<KimiK25MXFP4Experts>(
                runtime, model_config, rank_info);
            this->bind_child("experts", mxfp4_experts_);
    } else {
            experts_ = std::make_shared<std::remove_reference_t<decltype(*experts_)>>(
                runtime, model_config, rank_info, ep_config);
            this->bind_child("experts", experts_);
    }
        fused_moe_ = std::make_shared<std::remove_reference_t<decltype(*fused_moe_)>>(
            runtime, model_config, rank_info, ep_config, layer_idx);
        this->bind_child("fused_moe", fused_moe_);

    const size_t shared_intermediate_size = model_config->get<size_t>("moe_intermediate_size")
                                          * model_config->get_or<size_t>("n_shared_experts", 1);
    auto shared_config = make_kimi_mlp_config(model_config, shared_intermediate_size);
        shared_experts_ = std::make_shared<std::remove_reference_t<decltype(*shared_experts_)>>(
            runtime, shared_config, layer_idx, rank_info, attention_backend, ep_config);
        this->bind_child("shared_experts", shared_experts_);
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor KimiK25MoE::forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &hidden_states) const {
    const auto shape = hidden_states->shape();
    auto flattened = hidden_states->view({shape[0] * shape[1], shape[2]});
    auto [routing_weights, selected_experts] = gate_->forward(runtime, flattened);
    infinilm::layers::moe::TopKOutput topk_output{
        routing_weights,
        selected_experts,
        infinilm::runtime::Tensor(),
    };
    const auto &weights = uses_mxfp4_experts_
                            ? mxfp4_experts_->moe_weights()
                            : experts_->moe_weights();
    auto routed = fused_moe_->forward(runtime, flattened, topk_output, weights)->view(shape);
    {
        runtime.assertOwnerThread();
        if (!routed || runtime.device() != routed->device())
            throw std::invalid_argument("MulScalar runtime and input devices differ");
        auto scaled_routed = infinilm::runtime::Tensor::empty(runtime, routed->shape(), routed->dtype());
        infinilm::execution::submit<infini::ops::MulScalar>(
            runtime, infinilm::execution::RecordingPolicy::AllowHostReplay,
            infini::ops::MulScalar::DefaultConfig(runtime.device().type()),
            routed, static_cast<double>(routed_scaling_factor_), scaled_routed);
        INFINILM_TRACE_OP(MulScalar, scaled_routed);
        routed = std::move(scaled_routed);
    }
    const auto ops_other = shared_experts_->forward(runtime, hidden_states);
    auto ops_output = infinilm::runtime::Tensor::empty(runtime, routed->shape(), routed->dtype());
    infinilm::execution::submit<infini::ops::Add>(
        runtime, infini::ops::Add::DefaultConfig(runtime.device().type()), routed, ops_other, ops_output);
    INFINILM_TRACE_OP(Add, ops_output);
    return ops_output;
}

} // namespace infinilm::models::kimi_k25
