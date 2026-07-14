#include "fused_moe_experts.hpp"

#include "../../../global_state/global_state.hpp"
#include "../ep/ep_config.hpp"

#include <nlohmann/json.hpp>

#include <string>

namespace infinilm::layers::moe {
namespace {

bool use_compressed_tensors_quantization(const std::shared_ptr<infinilm::config::ModelConfig> &model_config) {
    const auto &config_json = model_config->get_config_json();
    const nlohmann::json *quant_config = nullptr;
    if (config_json.contains("quantization_config")) {
        quant_config = &config_json.at("quantization_config");
    } else if (config_json.contains("compression_config")) {
        quant_config = &config_json.at("compression_config");
    }
    if (quant_config == nullptr || quant_config->is_null()) {
        return false;
    }
    const std::string quant_method = quant_config->value("quant_method", "");
    return quant_method == "compressed-tensors" || quant_method == "compressed_tensors";
}

} // namespace

FusedMoeExperts::FusedMoeExperts(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                 const infinicore::Device &device) {
    num_experts_ = model_config->get<size_t>("num_experts");
    hidden_size_ = model_config->get<size_t>("hidden_size");
    const size_t intermediate_size = model_config->get<size_t>("moe_intermediate_size");
    const auto dtype = model_config->get_dtype();
    const bool use_w8a8 = use_compressed_tensors_quantization(model_config);
    ASSERT(num_experts_ > 0);

    const auto ep_config = make_ep_config();
    const auto expert_placement = make_expert_placement(ep_config, num_experts_);
    const size_t num_local_experts = expert_placement.local_num_experts;

    const engine::distributed::RankInfo &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    const size_t tp_rank = static_cast<size_t>(rank_info.tp_rank);
    const size_t tp_size = static_cast<size_t>(rank_info.tp_size);
    const bool ep_enabled = ep_config.backend != EPBackend::Disabled;
    if (ep_enabled) {
        intermediate_size_per_partition_ = intermediate_size;
    } else {
        ASSERT(intermediate_size % tp_size == 0);
        intermediate_size_per_partition_ = intermediate_size / tp_size;
    }
    const size_t expert_tp_rank = ep_enabled ? 0 : tp_rank;
    const size_t expert_tp_size = ep_enabled ? 1 : tp_size;

    const auto weight_dtype = use_w8a8 ? infinicore::DataType::I8 : dtype;
    w13_weight_ = infinicore::nn::Parameter(
        {num_local_experts, intermediate_size_per_partition_ * 2, hidden_size_},
        weight_dtype,
        device);
    w2_weight_ = infinicore::nn::Parameter(
        {num_local_experts, hidden_size_, intermediate_size_per_partition_},
        weight_dtype,
        device);
    this->register_parameter("w13_weight", w13_weight_);
    this->register_parameter("w2_weight", w2_weight_);
    if (use_w8a8) {
        w13_weight_scale_ = infinicore::nn::Parameter(
            {num_local_experts, intermediate_size_per_partition_ * 2, 1},
            infinicore::DataType::F32,
            device);
        w2_weight_scale_ = infinicore::nn::Parameter(
            {num_local_experts, hidden_size_, 1},
            infinicore::DataType::F32,
            device);
        this->register_parameter("w13_weight_scale", w13_weight_scale_);
        this->register_parameter("w2_weight_scale", w2_weight_scale_);
    }

    for (size_t local_expert = 0; local_expert < num_local_experts; ++local_expert) {
        const size_t global_expert = expert_placement.local_expert_start + local_expert;
        auto gate_weight = w13_weight_
                               ->narrow({{0, local_expert, 1}, {1, 0, intermediate_size_per_partition_}})
                               ->squeeze(0);
        auto up_weight = w13_weight_
                             ->narrow({{0, local_expert, 1}, {1, intermediate_size_per_partition_, intermediate_size_per_partition_}})
                             ->squeeze(0);
        auto down_weight = w2_weight_
                               ->narrow({{0, local_expert, 1}})
                               ->squeeze(0);

        const std::string prefix = std::to_string(global_expert) + ".";
        this->register_parameter(
            prefix + "gate_proj.weight",
            infinicore::nn::Parameter(gate_weight, 0, expert_tp_rank, expert_tp_size));
        this->register_parameter(
            prefix + "up_proj.weight",
            infinicore::nn::Parameter(up_weight, 0, expert_tp_rank, expert_tp_size));
        this->register_parameter(
            prefix + "down_proj.weight",
            infinicore::nn::Parameter(down_weight, 1, expert_tp_rank, expert_tp_size));
        if (use_w8a8) {
            auto gate_scale = w13_weight_scale_
                                  ->narrow({{0, local_expert, 1}, {1, 0, intermediate_size_per_partition_}})
                                  ->squeeze(0);
            auto up_scale = w13_weight_scale_
                                ->narrow({{0, local_expert, 1}, {1, intermediate_size_per_partition_, intermediate_size_per_partition_}})
                                ->squeeze(0);
            auto down_scale = w2_weight_scale_
                                  ->narrow({{0, local_expert, 1}})
                                  ->squeeze(0);
            this->register_parameter(
                prefix + "gate_proj.weight_scale",
                infinicore::nn::Parameter(gate_scale, 0, expert_tp_rank, expert_tp_size));
            this->register_parameter(
                prefix + "up_proj.weight_scale",
                infinicore::nn::Parameter(up_scale, 0, expert_tp_rank, expert_tp_size));
            this->register_parameter(
                prefix + "down_proj.weight_scale",
                infinicore::nn::Parameter(down_scale, -1, 0, 1));
        }
    }

    moe_weights_.packed_w13 = w13_weight_;
    moe_weights_.packed_w2 = w2_weight_;
    if (use_w8a8) {
        moe_weights_.packed_w13_scale = w13_weight_scale_;
        moe_weights_.packed_w2_scale = w2_weight_scale_;
    }
}

const MoeWeights &FusedMoeExperts::moe_weights() const {
    return moe_weights_;
}

} // namespace infinilm::layers::moe
