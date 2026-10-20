#include "../../weights/weights.hpp"
#include "../models_registry.hpp"
#include <exception>
#include "qwen3_moe_experts.hpp"

#include "../../config/model_config.hpp"
#include "../../global_state/global_state.hpp"
#include "runtime/runtime.hpp"
#include "../../layers/moe/ep/ep_config.hpp"
#include <stdexcept>

#include <optional>
#include <string>

namespace infinilm::models::qwen3_moe {

Qwen3MoeExperts::Qwen3MoeExperts(infinilm::runtime::Runtime &runtime, std::shared_ptr<config::ModelConfig> model_config,
    const engine::distributed::RankInfo &rank_info, const layers::moe::EPConfig &ep_config) {
    models::validate_selected_construction(model_config, runtime, rank_info, ep_config);
    try {

    const auto &dtype = model_config->get_dtype();
    const int tp_rank = rank_info.tp_rank;
    tp_size_ = rank_info.tp_size;

    num_experts_ = model_config->get<size_t>("num_experts");
    num_experts_per_tok_ = model_config->get<size_t>("num_experts_per_tok");
    hidden_size_ = model_config->get<size_t>("hidden_size");
    const size_t intermediate_size = model_config->get<size_t>("moe_intermediate_size");

    ASSERT((num_experts_ > 0) && (num_experts_per_tok_ > 0) && (num_experts_per_tok_ <= num_experts_));
    ASSERT(intermediate_size % static_cast<size_t>(tp_size_) == 0);
    intermediate_size_per_partition_ = intermediate_size / static_cast<size_t>(tp_size_);

    w1_ = infinilm::weights::Weight(runtime, {num_experts_, 2 * intermediate_size_per_partition_, hidden_size_}, dtype); this->bind_weight("w1", w1_);
    w2_ = infinilm::weights::Weight(runtime, {num_experts_, hidden_size_, intermediate_size_per_partition_}, dtype); this->bind_weight("w2", w2_);

    for (size_t expert = 0; expert < num_experts_; ++expert) {
        auto gate_weight = w1_
                               ->narrow({{0, expert, 1}, {1, 0, intermediate_size_per_partition_}})
                               ->squeeze(0);
        auto up_weight = w1_
                             ->narrow({{0, expert, 1}, {1, intermediate_size_per_partition_, intermediate_size_per_partition_}})
                             ->squeeze(0);
        auto down_weight = w2_
                               ->narrow({{0, expert, 1}})
                               ->squeeze(0);

        const std::string prefix = std::to_string(expert) + ".";
        this->bind_weight(
            prefix + "gate_proj.weight",
            infinilm::weights::Weight(gate_weight, 0, tp_rank, tp_size_));
        this->bind_weight(
            prefix + "up_proj.weight",
            infinilm::weights::Weight(up_weight, 0, tp_rank, tp_size_));
        this->bind_weight(
            prefix + "down_proj.weight",
            infinilm::weights::Weight(down_weight, 1, tp_rank, tp_size_));
    }
        if (runtime.device().type() != infinilm::runtime::Device::Type::kCpu) runtime.syncStream();
    } catch (...) {
        if (runtime.device().type() != infinilm::runtime::Device::Type::kCpu) {
            try { runtime.syncStream(); } catch (...) { std::terminate(); }
        }
        throw;
    }
}

infinilm::layers::moe::MoeWeights Qwen3MoeExperts::moe_weights(infinilm::runtime::Runtime &runtime) const {
    runtime.assertOwnerThread();
    if (w1_->device() != runtime.device() || w2_->device() != runtime.device()) throw std::invalid_argument("Legacy MoE packed weights device mismatch");
    return {static_cast<infinilm::runtime::Tensor>(w1_), static_cast<infinilm::runtime::Tensor>(w2_)};
}

} // namespace infinilm::models::qwen3_moe
