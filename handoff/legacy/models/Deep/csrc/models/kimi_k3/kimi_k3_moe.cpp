#include "execution/submit.hpp"
#include "../../weights/weights.hpp"
#include "kimi_k3_moe.hpp"

#include "../../global_state/global_state.hpp"
#include "runtime/runtime.hpp"

#include "../../engine/distributed/communication_group.hpp"
#include <infinilm::runtime/ops/fused_moe_mxfp4.hpp>
#include <infinilm::runtime/ops/situ_and_mul.hpp>
#include <exception>
#include <stdexcept>
#include <string>

namespace infinilm::models::kimi_k3 {
std::shared_ptr<infinilm::config::ModelConfig>
make_kimi_k3_subconfig(const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
                       size_t hidden_size,
                       size_t intermediate_size) {
    auto json = model_config->get_config_json();
    json["hidden_size"] = hidden_size;
    json["intermediate_size"] = intermediate_size;
    json["moe_intermediate_size"] = intermediate_size;
    return std::make_shared<infinilm::config::ModelConfig>(std::move(json));
}

KimiK3MLP::KimiK3MLP(
    infinilm::runtime::Runtime &runtime,
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
                     size_t intermediate_size,
    const engine::distributed::RankInfo &rank_info) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("KimiK3MLP construction requires eager Runtime");
    if (!model_config || rank_info.device != runtime.device() || rank_info.tp_size <= 0
        || rank_info.tp_rank < 0 || rank_info.tp_rank >= rank_info.tp_size
        || (rank_info.tp_size > 1 && !rank_info.comm))
        throw std::invalid_argument("Invalid explicit KimiK3MLP construction resources");
    try {
        device_ = runtime.device();
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const auto &dtype = model_config->get_dtype();
    situ_beta_ = model_config->get_or<float>("activation_situ_beta", 4.0f);
    situ_linear_beta_ = model_config->get_or<float>("activation_situ_linear_beta", 25.0f);
        auto register_fn = [this](const std::string &name, infinilm::weights::Weight parameter) {
            this->bind_weight(name, std::move(parameter));
    };
    gate_up_proj_ = std::make_shared<infinilm::layers::linear::GateUpParallelLinear>(
            runtime, hidden_size,
        intermediate_size,
        "gate_proj",
        "up_proj",
        register_fn,
        model_config->get_quantization_method(),
        false,
        dtype,
        rank_info);
        down_proj_ = std::make_shared<infinilm::layers::linear::RowParallelLinear>(
            runtime, intermediate_size, hidden_size, model_config->get_quantization_method(), false, dtype, rank_info.tp_rank, rank_info.tp_size, rank_info.comm);
        this->bind_child("down_proj", down_proj_);
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

void KimiK3MLP::process_weights_after_loading(infinilm::runtime::Runtime &runtime, bool pre_transpose) {
    runtime.assertOwnerThread();
    if (runtime.device() != device_)
        throw std::invalid_argument("KimiK3MLP: postload Runtime device differs");
    if (runtime.graphCaptureState() != infinilm::execution::graph::GraphManager::CaptureState::kInactive)
        throw std::logic_error("KimiK3MLP: postload requires inactive graph state");
    if (postload_attempted_)
        throw std::logic_error("KimiK3MLP: postload is single-use after mutation attempt");
    auto old_parameters = gate_up_proj_->state_dict();
    auto old_quantization = gate_up_proj_->get_quantization();
    postload_attempted_ = true;
    try {
        // Dense decoder MLP and MoE shared_experts both reach this hidden owner.
        // down_proj is registered and remains exclusively traversal-owned.
        gate_up_proj_->process_weights_after_loading(runtime, pre_transpose);
        runtime.syncStream();
    } catch (...) {
        const auto failure = std::current_exception();
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        std::rethrow_exception(failure);
    }
}

infinilm::runtime::Tensor KimiK3MLP::forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &hidden_states) const {
    auto input = hidden_states;
    auto [gate, up] = gate_up_proj_->forward_split(runtime, input);
    auto activated = infinilm::runtime::op::situ_and_mul(gate, up, situ_beta_, situ_linear_beta_);
    return down_proj_->forward(runtime, activated);
}

KimiK3Experts::KimiK3Experts(
    infinilm::runtime::Runtime &runtime,
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const engine::distributed::RankInfo &rank_info) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("KimiK3Experts construction requires eager Runtime");
    if (!model_config || rank_info.device != runtime.device() || rank_info.tp_size <= 0
        || rank_info.tp_rank < 0 || rank_info.tp_rank >= rank_info.tp_size
        || (rank_info.tp_size > 1 && !rank_info.comm))
        throw std::invalid_argument("Invalid explicit KimiK3Experts construction resources");
    try {
        num_experts_ = model_config->get<size_t>("num_experts");
        hidden_size_ = model_config->get<size_t>("hidden_size");
        device_ = runtime.device();
    tp_rank_ = static_cast<size_t>(rank_info.tp_rank);
    tp_size_ = static_cast<size_t>(rank_info.tp_size);
    const size_t intermediate_size = model_config->get<size_t>("moe_intermediate_size");
    if (intermediate_size % tp_size_ != 0) {
        throw std::runtime_error("KimiK3Experts: moe_intermediate_size must be divisible by tp_size");
    }
    local_intermediate_size_ = intermediate_size / tp_size_;
        register_mxfp4_experts(runtime);
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

void KimiK3Experts::register_mxfp4_experts(infinilm::runtime::Runtime &runtime) {
    if (hidden_size_ % 32 != 0 || local_intermediate_size_ % 32 != 0) {
        throw std::runtime_error("KimiK3Experts: MXFP4 dimensions must be divisible by 32");
    }
    auto w13 = infinilm::runtime::Tensor::empty(
        runtime, {num_experts_, 2 * local_intermediate_size_, hidden_size_ / 2},
        infinilm::runtime::DataType::kUInt8);
    auto w13_scale = infinilm::runtime::Tensor::empty(
        runtime, {num_experts_, 2 * local_intermediate_size_, hidden_size_ / 32},
        infinilm::runtime::DataType::kUInt8);
    auto w2 = infinilm::runtime::Tensor::empty(
        runtime, {num_experts_, hidden_size_, local_intermediate_size_ / 2},
        infinilm::runtime::DataType::kUInt8);
    auto w2_scale = infinilm::runtime::Tensor::empty(
        runtime, {num_experts_, hidden_size_, local_intermediate_size_ / 32},
        infinilm::runtime::DataType::kUInt8);

    auto register_packed = [&](const std::string &name,
                               const infinilm::runtime::Tensor &storage,
                               size_t expert,
                               size_t row_start,
                               size_t row_count,
                               size_t tp_dim) {
        auto expert_view = storage
                               ->narrow({{0, expert, 1}, {1, row_start, row_count}})
                               ->squeeze(0);
        this->bind_weight(
            name,
            infinilm::weights::Weight(expert_view, tp_dim, tp_rank_, tp_size_));
    };
    for (size_t expert = 0; expert < num_experts_; ++expert) {
        const std::string prefix = std::to_string(expert) + ".";
        register_packed(prefix + "w1.weight_packed", w13, expert,
                        0, local_intermediate_size_, 0);
        register_packed(prefix + "w1.weight_scale", w13_scale, expert,
                        0, local_intermediate_size_, 0);
        register_packed(prefix + "w3.weight_packed", w13, expert,
                        local_intermediate_size_, local_intermediate_size_, 0);
        register_packed(prefix + "w3.weight_scale", w13_scale, expert,
                        local_intermediate_size_, local_intermediate_size_, 0);
        register_packed(prefix + "w2.weight_packed", w2, expert,
                        0, hidden_size_, 1);
        register_packed(prefix + "w2.weight_scale", w2_scale, expert,
                        0, hidden_size_, 1);
    }

    mxfp4_weights_.packed_w13 = std::move(w13);
    mxfp4_weights_.w13_scale = std::move(w13_scale);
    mxfp4_weights_.packed_w2 = std::move(w2);
    mxfp4_weights_.w2_scale = std::move(w2_scale);
}

const KimiK3Mxfp4MoeWeights &KimiK3Experts::mxfp4_weights() const {
    return mxfp4_weights_;
}

KimiK3MoE::KimiK3MoE(
    infinilm::runtime::Runtime &runtime,
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
                     size_t layer_idx,
    const engine::distributed::RankInfo &rank_info,
    backends::AttentionBackend attention_backend,
    const layers::moe::EPConfig &ep_config) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("KimiK3MoE construction requires eager Runtime");
    if (!model_config || rank_info.device != runtime.device() || rank_info.tp_size <= 0
        || rank_info.tp_rank < 0 || rank_info.tp_rank >= rank_info.tp_size
        || (rank_info.tp_size > 1 && !rank_info.comm))
        throw std::invalid_argument("Invalid explicit KimiK3MoE construction resources");
    try {
    (void)layer_idx;
        (void)attention_backend;
        (void)ep_config; // Preserve the existing custom TP expert path; no new EP policy.
    const auto &dtype = model_config->get_dtype();
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t routed_hidden_size = model_config->get<size_t>("routed_expert_hidden_size");
    const size_t expert_intermediate_size = model_config->get<size_t>("moe_intermediate_size");
    const size_t shared_intermediate_size = expert_intermediate_size
                                          * model_config->get<size_t>("num_shared_experts");
    const double eps = model_config->get<double>("rms_norm_eps");
    tp_size_ = rank_info.tp_size;
    communicator_ = rank_info.comm;

        gate_ = std::make_shared<std::remove_reference_t<decltype(*gate_)>>(
            runtime, model_config);
        this->bind_child("gate", gate_);
        routed_expert_down_proj_ = std::make_shared<std::remove_reference_t<decltype(*routed_expert_down_proj_)>>(
            runtime, hidden_size, routed_hidden_size, false, dtype);
        this->bind_child("routed_expert_down_proj", routed_expert_down_proj_);
    auto expert_config = make_kimi_k3_subconfig(
        model_config, routed_hidden_size, expert_intermediate_size);
        experts_ = std::make_shared<std::remove_reference_t<decltype(*experts_)>>(
            runtime, expert_config, rank_info);
        this->bind_child("experts", experts_);
        routed_expert_norm_ = std::make_shared<std::remove_reference_t<decltype(*routed_expert_norm_)>>(
            runtime, routed_hidden_size, eps, dtype);
        this->bind_child("routed_expert_norm", routed_expert_norm_);
        routed_expert_up_proj_ = std::make_shared<std::remove_reference_t<decltype(*routed_expert_up_proj_)>>(
            runtime, routed_hidden_size, hidden_size, false, dtype);
        this->bind_child("routed_expert_up_proj", routed_expert_up_proj_);
        shared_experts_ = std::make_shared<std::remove_reference_t<decltype(*shared_experts_)>>(
            runtime, model_config, shared_intermediate_size, rank_info);
        this->bind_child("shared_experts", shared_experts_);
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor KimiK3MoE::forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &hidden_states) const {
    const auto shape = hidden_states->shape();
    auto flattened = hidden_states->view({shape[0] * shape[1], shape[2]});
    auto [routing_weights, selected_experts] = gate_->forward(runtime, flattened);
    auto routed_input = routed_expert_down_proj_->forward(runtime, flattened);
    const auto &weights = experts_->mxfp4_weights();
    auto routed = infinilm::runtime::op::fused_moe_mxfp4(
        routed_input,
        selected_experts,
        routing_weights,
        weights.packed_w13,
        weights.w13_scale,
        weights.packed_w2,
        weights.w2_scale,
        infinilm::runtime::op::FusedMoeActivation::Situglu);
    if (tp_size_ > 1 && communicator_ != nullptr) {
        infinilm::engine::distributed::allreduce_(runtime,
            routed, routed, INFINICCL_SUM, communicator_);
    }
    routed = routed_expert_norm_->forward(runtime, routed);
    routed = routed_expert_up_proj_->forward(runtime, routed)->view(shape);
    auto shared = shared_experts_->forward(runtime, hidden_states);
    auto ops_output = infinilm::runtime::Tensor::empty(runtime, routed->shape(), routed->dtype());
    infinilm::execution::submit<infini::ops::Add>(
        runtime, infini::ops::Add::DefaultConfig(runtime.device().type()), routed, shared, ops_output);
    INFINILM_TRACE_OP(Add, ops_output);
    return ops_output;
}

} // namespace infinilm::models::kimi_k3
