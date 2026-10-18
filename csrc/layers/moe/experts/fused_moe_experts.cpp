#include "fused_moe_experts.hpp"
#include "../../../weights/weights.hpp"

#include "../../../engine/distributed/communication_group.hpp"
#include "../../../global_state/global_state.hpp"
#include "../ep/ep_config.hpp"
#include "runtime/runtime.hpp"

#include <exception>
#include <limits>
#include <stdexcept>
#include <string>

namespace infinilm::layers::moe {

FusedMoeExperts::FusedMoeExperts(infinilm::runtime::Runtime &runtime,
                                 std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                 const engine::distributed::RankInfo &rank_info,
                                 const EPConfig &ep_config) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) {
        throw std::logic_error("FusedMoeExperts construction requires eager Runtime");
    }
    if (!model_config) {
        throw std::invalid_argument("FusedMoeExperts construction requires configuration");
    }
    if (rank_info.device != runtime.device() || rank_info.tp_size <= 0 || rank_info.tp_rank < 0 || rank_info.tp_rank >= rank_info.tp_size
        || rank_info.pp_size <= 0 || rank_info.pp_stage < 0 || rank_info.pp_stage >= rank_info.pp_size
        || rank_info.world_size <= 0 || rank_info.world_rank < 0 || rank_info.world_rank >= rank_info.world_size
        || static_cast<int64_t>(rank_info.world_size) != static_cast<int64_t>(rank_info.tp_size) * rank_info.pp_size
        || static_cast<int64_t>(rank_info.world_rank) != static_cast<int64_t>(rank_info.pp_stage) * rank_info.tp_size + rank_info.tp_rank
        || (rank_info.tp_size > 1 && !rank_info.comm) || (rank_info.pp_size > 1 && !rank_info.world_comm)) {
        throw std::invalid_argument("FusedMoeExperts construction rank topology is invalid");
    }
    for (const char *key : {"num_experts", "hidden_size", "moe_intermediate_size"}) {
        if (model_config->get<int64_t>(key) <= 0) {
            throw std::invalid_argument("FusedMoeExperts construction requires positive dimensions");
        }
    }
    num_experts_ = model_config->get<size_t>("num_experts");
    hidden_size_ = model_config->get<size_t>("hidden_size");
    const size_t intermediate_size = model_config->get<size_t>("moe_intermediate_size");
    const auto dtype = model_config->get_dtype();
    const size_t tp_rank = static_cast<size_t>(rank_info.tp_rank), tp_size = static_cast<size_t>(rank_info.tp_size);
    const bool ep_enabled = ep_config.backend != EPBackend::Disabled;
    if (ep_enabled && (ep_config.ep_size != tp_size || ep_config.ep_rank != tp_rank)) {
        throw std::invalid_argument("MoE EP size/rank must match its existing TP communication group");
    }
    if (ep_config.backend != EPBackend::Disabled && ep_config.backend != EPBackend::AllGatherReduceScatter
        && ep_config.backend != EPBackend::LocalAllReduce && ep_config.backend != EPBackend::DeepEP) {
        throw std::invalid_argument("Unsupported MoE EP backend");
    }
    const auto expert_placement = make_expert_placement(ep_config, num_experts_);
    const size_t num_local_experts = expert_placement.local_num_experts;
    if (!ep_enabled && intermediate_size % tp_size != 0) {
        throw std::invalid_argument("moe_intermediate_size must be divisible by tensor parallel world size");
    }
    intermediate_size_per_partition_ = ep_enabled ? intermediate_size : intermediate_size / tp_size;
    const size_t expert_tp_rank = ep_enabled ? 0 : tp_rank;
    const size_t expert_tp_size = ep_enabled ? 1 : tp_size;
    const size_t limit = std::numeric_limits<size_t>::max(), bytes = infinilm::runtime::dsize(dtype);
    if (!bytes || intermediate_size_per_partition_ > limit / 2
        || hidden_size_ > limit / (2 * intermediate_size_per_partition_)
        || num_local_experts > limit / (2 * intermediate_size_per_partition_ * hidden_size_)
        || num_local_experts * (2 * intermediate_size_per_partition_ * hidden_size_) > limit / bytes) {
        throw std::length_error("FusedMoeExperts packed parameter size overflow");
    }
    device_ = runtime.device();
    try {
        // Members retain both packed owners before alias registration can throw.
        w13_weight_ = infinilm::weights::Weight(runtime,
                                                {num_local_experts, intermediate_size_per_partition_ * 2, hidden_size_}, dtype);
        w2_weight_ = infinilm::weights::Weight(runtime,
                                               {num_local_experts, hidden_size_, intermediate_size_per_partition_}, dtype);
        this->bind_derived_weight("w13_weight", w13_weight_);
        this->bind_derived_weight("w2_weight", w2_weight_);
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
            this->bind_weight(prefix + "gate_proj.weight",
                              infinilm::weights::Weight(gate_weight, 0, expert_tp_rank, expert_tp_size));
            this->bind_weight(prefix + "up_proj.weight",
                              infinilm::weights::Weight(up_weight, 0, expert_tp_rank, expert_tp_size));
            this->bind_weight(prefix + "down_proj.weight",
                              infinilm::weights::Weight(down_weight, 1, expert_tp_rank, expert_tp_size));
        }
        moe_weights_.packed_w13 = w13_weight_;
        moe_weights_.packed_w2 = w2_weight_;
    } catch (...) {
        try {
            runtime.syncStream();
        } catch (...) { std::terminate(); }
        throw;
    }
}

const MoeWeights &FusedMoeExperts::moe_weights() const {
    return moe_weights_;
}

} // namespace infinilm::layers::moe
