#include "fused_moe_experts.hpp"

#include "../../../global_state/global_state.hpp"
#include "../ep/ep_config.hpp"

#include "infinicore/ops/moe_w16a16_marlin.hpp"
#include "infinicore/ops/moe_w8a8_marlin.hpp"

#include <spdlog/spdlog.h>

#include <stdexcept>
#include <string>

namespace infinilm::layers::moe {

FusedMoeExperts::FusedMoeExperts(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                 const infinicore::Device &device) {
    device_ = device;
    num_experts_ = model_config->get<size_t>("num_experts");
    hidden_size_ = model_config->get<size_t>("hidden_size");
    const size_t intermediate_size = model_config->get<size_t>("moe_intermediate_size");
    const auto dtype = model_config->get_dtype();
    const auto moe_weight_method = model_config->get_moe_weight_method(device);
    enable_hygon_w16a16_marlin_ = model_config->is_moe_w16a16_marlin_enabled(device);
    enable_hygon_w8a8_marlin_ = model_config->is_moe_w8a8_marlin_enabled(device);
    if (enable_hygon_w16a16_marlin_ && enable_hygon_w8a8_marlin_) {
        throw std::runtime_error("Only one Hygon MoE Marlin weight method can be enabled");
    }
    if (moe_weight_method != "dense" &&
        !enable_hygon_w16a16_marlin_ && !enable_hygon_w8a8_marlin_) {
        throw std::runtime_error("Unsupported MoE weight method: " + moe_weight_method);
    }
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

    const auto expert_weight_dtype = enable_hygon_w8a8_marlin_ ? infinicore::DataType::I8 : dtype;
    w13_weight_ = infinicore::nn::Parameter(
        {num_local_experts, intermediate_size_per_partition_ * 2, hidden_size_},
        expert_weight_dtype,
        device);
    w2_weight_ = infinicore::nn::Parameter(
        {num_local_experts, hidden_size_, intermediate_size_per_partition_},
        expert_weight_dtype,
        device);
    this->register_parameter("w13_weight", w13_weight_);
    this->register_parameter("w2_weight", w2_weight_);

    if (enable_hygon_w8a8_marlin_) {
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

        if (enable_hygon_w8a8_marlin_) {
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
                infinicore::nn::Parameter(down_scale));
        }
    }

    moe_weights_.packed_w13 = w13_weight_;
    moe_weights_.packed_w2 = w2_weight_;
    moe_weights_.backend = MoeWeightBackend::Dense;
}

void FusedMoeExperts::process_weights_after_loading() {
    if (enable_hygon_w8a8_marlin_ && !w8a8_marlin_packed_) {
        if (device_.getType() != infinicore::Device::Type::HYGON) {
            throw std::runtime_error("slimquant_marlin MoE weight method is only supported on HYGON");
        }
        const auto ep_config = make_ep_config();
        if (ep_config.backend != EPBackend::Disabled) {
            throw std::runtime_error("slimquant_marlin MoE weight method currently supports TP-split experts only; disable MoE EP");
        }
        if (!w13_weight_ || !w2_weight_ || !w13_weight_scale_ || !w2_weight_scale_) {
            throw std::runtime_error("slimquant_marlin MoE weight method requires loaded int8 w13/w2 weights and scales");
        }
        if (w13_weight_->dtype() != infinicore::DataType::I8 ||
            w2_weight_->dtype() != infinicore::DataType::I8 ||
            w13_weight_scale_->dtype() != infinicore::DataType::F32 ||
            w2_weight_scale_->dtype() != infinicore::DataType::F32) {
            throw std::runtime_error("slimquant_marlin MoE weight method requires int8 weights and fp32 weight scales");
        }
        if (hidden_size_ % 64 != 0 || intermediate_size_per_partition_ % 64 != 0) {
            throw std::runtime_error("slimquant_marlin MoE weight method requires hidden/intermediate sizes divisible by 64");
        }

        spdlog::debug(
            "Packing MoE weights with Hygon W8A8 slimquant Marlin layout: experts={}, hidden={}, intermediate_per_partition={}",
            w13_weight_->size(0), hidden_size_, intermediate_size_per_partition_);

        auto packed_w13 = infinicore::op::moe_w8a8_marlin_pack(w13_weight_);
        auto packed_w2 = infinicore::op::moe_w8a8_marlin_pack(w2_weight_);

        parameters_.clear();
        w13_weight_ = infinicore::nn::Parameter(packed_w13);
        w2_weight_ = infinicore::nn::Parameter(packed_w2);
        w13_weight_scale_ = infinicore::nn::Parameter(w13_weight_scale_);
        w2_weight_scale_ = infinicore::nn::Parameter(w2_weight_scale_);
        this->register_parameter("w13_weight", w13_weight_);
        this->register_parameter("w2_weight", w2_weight_);
        this->register_parameter("w13_weight_scale", w13_weight_scale_);
        this->register_parameter("w2_weight_scale", w2_weight_scale_);

        moe_weights_.packed_w13 = w13_weight_;
        moe_weights_.packed_w2 = w2_weight_;
        moe_weights_.packed_w13_scale = w13_weight_scale_;
        moe_weights_.packed_w2_scale = w2_weight_scale_;
        moe_weights_.backend = MoeWeightBackend::HygonW8A8Marlin;
        w8a8_marlin_packed_ = true;
        return;
    }

    if (!enable_hygon_w16a16_marlin_ || w16a16_marlin_packed_) {
        return;
    }
    if (device_.getType() != infinicore::Device::Type::HYGON) {
        throw std::runtime_error("w16a16_marlin MoE weight method is only supported on HYGON");
    }

    const auto ep_config = make_ep_config();
    if (ep_config.backend != EPBackend::Disabled) {
        throw std::runtime_error("w16a16_marlin MoE weight method currently supports TP-split experts only; disable MoE EP");
    }
    if (!w13_weight_ || !w2_weight_) {
        throw std::runtime_error("w16a16_marlin MoE weight method requires loaded dense w13/w2 weights");
    }
    if (w13_weight_->dtype() != infinicore::DataType::F16 &&
        w13_weight_->dtype() != infinicore::DataType::BF16) {
        throw std::runtime_error("w16a16_marlin MoE weight method requires FP16 or BF16 weights");
    }
    if (hidden_size_ % 32 != 0 || intermediate_size_per_partition_ % 16 != 0 ||
        (intermediate_size_per_partition_ * 2) % 32 != 0) {
        throw std::runtime_error("w16a16_marlin MoE weight method requires aligned hidden/intermediate sizes");
    }

    spdlog::debug(
        "Packing MoE weights with Hygon W16A16 Marlin layout: experts={}, hidden={}, intermediate_per_partition={}",
        w13_weight_->size(0), hidden_size_, intermediate_size_per_partition_);

    auto packed_w13 = infinicore::op::moe_w16a16_marlin_pack(w13_weight_);
    auto packed_w2 = infinicore::op::moe_w16a16_marlin_pack(w2_weight_);

    parameters_.clear();
    w13_weight_ = infinicore::nn::Parameter(packed_w13);
    w2_weight_ = infinicore::nn::Parameter(packed_w2);
    this->register_parameter("w13_weight", w13_weight_);
    this->register_parameter("w2_weight", w2_weight_);

    moe_weights_.packed_w13 = w13_weight_;
    moe_weights_.packed_w2 = w2_weight_;
    moe_weights_.backend = MoeWeightBackend::HygonW16A16Marlin;
    w16a16_marlin_packed_ = true;
}

const MoeWeights &FusedMoeExperts::moe_weights() const {
    return moe_weights_;
}

} // namespace infinilm::layers::moe
