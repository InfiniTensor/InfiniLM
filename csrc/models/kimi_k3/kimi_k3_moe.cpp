#include "kimi_k3_moe.hpp"

#include "../../global_state/global_state.hpp"

#include <infinicore/ops/add.hpp>
#include <infinicore/ops/distributed/allreduce.hpp>
#include <infinicore/ops/fused_moe_mxfp4.hpp>
#include <infinicore/ops/fused_moe_w4a8.hpp>
#include <infinicore/ops/situ_and_mul.hpp>
#include <infinicore/ops/w4a8_moe_shuffle.hpp>
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

KimiK3MLP::KimiK3MLP(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                     size_t intermediate_size,
                     const infinicore::Device &device) {
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const auto &dtype = model_config->get_dtype();
    const auto &rank_info = global_state::get_tensor_model_parallel_rank_info();
    situ_beta_ = model_config->get_or<float>("activation_situ_beta", 4.0f);
    situ_linear_beta_ = model_config->get_or<float>("activation_situ_linear_beta", 25.0f);
    auto register_fn = [this](const std::string &name, infinicore::nn::Parameter parameter) {
        this->register_parameter(name, std::move(parameter));
    };
    gate_up_proj_ = std::make_shared<infinilm::layers::linear::GateUpParallelLinear>(
        hidden_size,
        intermediate_size,
        "gate_proj",
        "up_proj",
        register_fn,
        model_config->get_quantization_method(),
        false,
        dtype,
        device,
        rank_info);
    down_proj_ = this->register_module<infinilm::layers::linear::RowParallelLinear>(
        "down_proj",
        intermediate_size,
        hidden_size,
        model_config->get_quantization_method(),
        false,
        dtype,
        device,
        rank_info.tp_rank,
        rank_info.tp_size,
        rank_info.comm);
}

infinicore::Tensor KimiK3MLP::forward(const infinicore::Tensor &hidden_states) const {
    auto input = hidden_states;
    auto [gate, up] = gate_up_proj_->forward_split(input);
    auto activated = infinicore::op::situ_and_mul(gate, up, situ_beta_, situ_linear_beta_);
    return down_proj_->forward(activated);
}

KimiK3Experts::KimiK3Experts(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                             const infinicore::Device &device)
    : num_experts_(model_config->get<size_t>("num_experts")),
      hidden_size_(model_config->get<size_t>("hidden_size")),
      device_(device) {
    const auto &rank_info = global_state::get_tensor_model_parallel_rank_info();
    tp_rank_ = static_cast<size_t>(rank_info.tp_rank);
    tp_size_ = static_cast<size_t>(rank_info.tp_size);
    const size_t intermediate_size = model_config->get<size_t>("moe_intermediate_size");
    if (intermediate_size % tp_size_ != 0) {
        throw std::runtime_error("KimiK3Experts: moe_intermediate_size must be divisible by tp_size");
    }
    local_intermediate_size_ = intermediate_size / tp_size_;
    const std::string quant_method = model_config->get_or<std::string>(
        "routed_expert_quant_method", "mxfp4");
    if (quant_method == "slimquant_w4a8") {
        quantization_ = KimiK3ExpertQuantization::W4A8;
        register_w4a8_experts();
    } else if (quant_method == "mxfp4") {
        register_mxfp4_experts();
    } else {
        throw std::runtime_error(
            "KimiK3Experts: unsupported routed expert quantization " + quant_method);
    }
}

void KimiK3Experts::register_mxfp4_experts() {
    if (hidden_size_ % 32 != 0 || local_intermediate_size_ % 32 != 0) {
        throw std::runtime_error("KimiK3Experts: MXFP4 dimensions must be divisible by 32");
    }
    auto w13 = infinicore::Tensor::empty(
        {num_experts_, 2 * local_intermediate_size_, hidden_size_ / 2},
        infinicore::DataType::U8, device_);
    auto w13_scale = infinicore::Tensor::empty(
        {num_experts_, 2 * local_intermediate_size_, hidden_size_ / 32},
        infinicore::DataType::U8, device_);
    auto w2 = infinicore::Tensor::empty(
        {num_experts_, hidden_size_, local_intermediate_size_ / 2},
        infinicore::DataType::U8, device_);
    auto w2_scale = infinicore::Tensor::empty(
        {num_experts_, hidden_size_, local_intermediate_size_ / 32},
        infinicore::DataType::U8, device_);

    auto register_packed = [&](const std::string &name,
                               const infinicore::Tensor &storage,
                               size_t expert,
                               size_t row_start,
                               size_t row_count,
                               size_t tp_dim) {
        auto expert_view = storage
                               ->narrow({{0, expert, 1}, {1, row_start, row_count}})
                               ->squeeze(0);
        this->register_parameter(
            name,
            infinicore::nn::Parameter(expert_view, tp_dim, tp_rank_, tp_size_));
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

    weights_.packed_w13 = std::move(w13);
    weights_.w13_scale = std::move(w13_scale);
    weights_.packed_w2 = std::move(w2);
    weights_.w2_scale = std::move(w2_scale);
}

void KimiK3Experts::register_w4a8_experts() {
    if (hidden_size_ % 64 != 0 || local_intermediate_size_ % 64 != 0) {
        throw std::runtime_error("KimiK3Experts: W4A8 dimensions must be divisible by 64");
    }
    auto w13 = infinicore::Tensor::empty(
        {num_experts_, 2 * local_intermediate_size_, hidden_size_ / 2},
        infinicore::DataType::I8, device_);
    auto w13_scale = infinicore::Tensor::empty(
        {num_experts_, 2 * local_intermediate_size_, 1},
        infinicore::DataType::F32, device_);
    auto w2 = infinicore::Tensor::empty(
        {num_experts_, hidden_size_, local_intermediate_size_ / 2},
        infinicore::DataType::I8, device_);
    auto w2_scale = infinicore::Tensor::empty(
        {num_experts_, hidden_size_, 1},
        infinicore::DataType::F32, device_);

    for (size_t expert = 0; expert < num_experts_; ++expert) {
        const std::string prefix = std::to_string(expert) + ".";
        auto register_rows = [&](const std::string &name,
                                 const infinicore::Tensor &storage,
                                 size_t row_start,
                                 size_t row_count) {
            auto view = storage
                            ->narrow({{0, expert, 1}, {1, row_start, row_count}})
                            ->squeeze(0);
            this->register_parameter(
                name, infinicore::nn::Parameter(view, 0, tp_rank_, tp_size_));
        };
        register_rows(prefix + "w1.weight", w13, 0, local_intermediate_size_);
        register_rows(prefix + "w1.weight_scale", w13_scale, 0, local_intermediate_size_);
        register_rows(prefix + "w3.weight", w13, local_intermediate_size_, local_intermediate_size_);
        register_rows(prefix + "w3.weight_scale", w13_scale, local_intermediate_size_, local_intermediate_size_);

        auto w2_view = w2->narrow({{0, expert, 1}})->squeeze(0);
        this->register_parameter(
            prefix + "w2.weight",
            infinicore::nn::Parameter(w2_view, 1, tp_rank_, tp_size_));
        auto w2_scale_view = w2_scale->narrow({{0, expert, 1}})->squeeze(0);
        this->register_parameter(
            prefix + "w2.weight_scale",
            infinicore::nn::Parameter(w2_scale_view));
    }

    weights_.packed_w13 = std::move(w13);
    weights_.w13_scale = std::move(w13_scale);
    weights_.packed_w2 = std::move(w2);
    weights_.w2_scale = std::move(w2_scale);
}

void KimiK3Experts::process_weights_after_loading() {
    if (quantization_ != KimiK3ExpertQuantization::W4A8
        || weights_are_aiter_shuffled_) {
        return;
    }

    auto shuffle_experts = [&](const infinicore::Tensor &packed_weights) {
        auto scratch = infinicore::Tensor::empty(
            {packed_weights->size(1), packed_weights->size(2)},
            packed_weights->dtype(), packed_weights->device());
        for (size_t expert = 0; expert < num_experts_; ++expert) {
            auto expert_weights = packed_weights->narrow({{0, expert, 1}})->squeeze(0);
            infinicore::op::w4a8_moe_shuffle_(scratch, expert_weights);
            expert_weights->copy_from(scratch);
        }
    };
    shuffle_experts(weights_.packed_w13);
    shuffle_experts(weights_.packed_w2);

    weights_are_aiter_shuffled_ = true;
}

const KimiK3MoeWeights &KimiK3Experts::weights() const {
    return weights_;
}

KimiK3MoE::KimiK3MoE(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                     size_t layer_idx,
                     const infinicore::Device &device) {
    (void)layer_idx;
    const auto &dtype = model_config->get_dtype();
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t routed_hidden_size = model_config->get<size_t>("routed_expert_hidden_size");
    const size_t expert_intermediate_size = model_config->get<size_t>("moe_intermediate_size");
    const size_t shared_intermediate_size = expert_intermediate_size
                                          * model_config->get<size_t>("num_shared_experts");
    const double eps = model_config->get<double>("rms_norm_eps");
    const auto &rank_info = global_state::get_tensor_model_parallel_rank_info();
    tp_size_ = rank_info.tp_size;
    communicator_ = rank_info.comm;

    INFINICORE_NN_MODULE_INIT(gate, model_config, device);
    INFINICORE_NN_MODULE_INIT(routed_expert_down_proj,
                              hidden_size, routed_hidden_size, false, dtype, device);
    auto expert_config = make_kimi_k3_subconfig(
        model_config, routed_hidden_size, expert_intermediate_size);
    INFINICORE_NN_MODULE_INIT(experts, expert_config, device);
    INFINICORE_NN_MODULE_INIT(routed_expert_norm, routed_hidden_size, eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(routed_expert_up_proj,
                              routed_hidden_size, hidden_size, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(shared_experts, model_config, shared_intermediate_size, device);
}

infinicore::Tensor KimiK3MoE::forward(const infinicore::Tensor &hidden_states) const {
    const auto shape = hidden_states->shape();
    auto flattened = hidden_states->view({shape[0] * shape[1], shape[2]});
    auto [routing_weights, selected_experts] = gate_->forward(flattened);
    auto routed_input = routed_expert_down_proj_->forward(flattened);
    const auto &weights = experts_->weights();
    infinicore::Tensor routed;
    if (experts_->quantization() == KimiK3ExpertQuantization::W4A8) {
        routed = infinicore::op::fused_moe_w4a8(
            routed_input, selected_experts, routing_weights,
            weights.packed_w13, weights.w13_scale,
            weights.packed_w2, weights.w2_scale,
            infinicore::op::FusedMoeActivation::Situglu,
            experts_->weights_are_aiter_shuffled());
    } else {
        routed = infinicore::op::fused_moe_mxfp4(
            routed_input, selected_experts, routing_weights,
            weights.packed_w13, weights.w13_scale,
            weights.packed_w2, weights.w2_scale,
            infinicore::op::FusedMoeActivation::Situglu);
    }
    if (tp_size_ > 1 && communicator_ != nullptr) {
        infinicore::op::distributed::allreduce_(
            routed, routed, INFINICCL_SUM, communicator_);
    }
    routed = routed_expert_norm_->forward(routed);
    routed = routed_expert_up_proj_->forward(routed)->view(shape);
    auto shared = shared_experts_->forward(hidden_states);
    return infinicore::op::add(routed, shared);
}

} // namespace infinilm::models::kimi_k3
