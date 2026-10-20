#include "fused_moe.hpp"
#include <exception>

#include "dispatcher/dispatcher_factory.hpp"
#include "ep/ep_config.hpp"
#include "runner/cuda_fused_moe_runner.hpp"

#include "runtime/runtime.hpp"

#include <stdexcept>

namespace infinilm::layers::moe {

FusedMoE::FusedMoE(infinilm::runtime::Runtime &runtime, std::shared_ptr<config::ModelConfig> model_config,
                   const engine::distributed::RankInfo &rank_info, const layers::moe::EPConfig &ep_config, size_t layer_id) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) {
        throw std::logic_error("FusedMoE construction requires eager Runtime");
    }
    if (!model_config) {
        throw std::invalid_argument("FusedMoE construction requires configuration");
    }
    if (rank_info.device != runtime.device() || rank_info.tp_size <= 0 || rank_info.tp_rank < 0 || rank_info.tp_rank >= rank_info.tp_size
        || rank_info.pp_size <= 0 || rank_info.pp_stage < 0 || rank_info.pp_stage >= rank_info.pp_size
        || rank_info.world_size <= 0 || rank_info.world_rank < 0 || rank_info.world_rank >= rank_info.world_size
        || static_cast<int64_t>(rank_info.world_size) != static_cast<int64_t>(rank_info.tp_size) * rank_info.pp_size
        || static_cast<int64_t>(rank_info.world_rank) != static_cast<int64_t>(rank_info.pp_stage) * rank_info.tp_size + rank_info.tp_rank
        || (rank_info.tp_size > 1 && !rank_info.comm) || (rank_info.pp_size > 1 && !rank_info.world_comm)) {
        throw std::invalid_argument("FusedMoE construction rank topology is invalid");
    }
    for (const char *key : {"num_experts", "hidden_size", "moe_intermediate_size"}) {
        if (model_config->get<int64_t>(key) <= 0) {
            throw std::invalid_argument("FusedMoE construction requires positive dimensions");
        }
    }
    if (model_config->get_or<int64_t>("moe_align_block_size", 16) <= 0) {
        throw std::invalid_argument("FusedMoE requires positive moe_align_block_size");
    }
    if (ep_config.backend != EPBackend::Disabled
        && (ep_config.ep_size != static_cast<size_t>(rank_info.tp_size)
            || ep_config.ep_rank != static_cast<size_t>(rank_info.tp_rank))) {
        throw std::invalid_argument("MoE EP size/rank must match its existing TP communication group");
    }
    try {

        (void)layer_id;

        const size_t num_experts = model_config->get<size_t>("num_experts");
        const ExpertPlacement expert_placement = make_expert_placement(ep_config, num_experts);
        const size_t hidden_size = model_config->get<size_t>("hidden_size");
        const size_t tp_size = static_cast<size_t>(rank_info.tp_size);
        const size_t intermediate_size = model_config->get<size_t>("moe_intermediate_size");
        size_t intermediate_size_per_partition = intermediate_size;
        if (ep_config.backend == EPBackend::Disabled) {
            if (intermediate_size % tp_size != 0) {
                throw std::runtime_error("moe_intermediate_size must be divisible by tensor parallel world size");
            }
            intermediate_size_per_partition = intermediate_size / tp_size;
        }

        dispatcher_ = make_dispatcher(ep_config, num_experts, rank_info);
        runner_ = std::make_shared<CudaFusedMoeRunner>(
            expert_placement.local_num_experts,
            hidden_size,
            intermediate_size_per_partition,
            model_config->get_or<size_t>("moe_align_block_size", 16));
        runtime_borrowed_ = &runtime;
        dispatcher_->initialize(runtime, workspace_);
        if (runtime.device().type() != infinilm::runtime::Device::Type::kCpu) {
            runtime.syncStream();
        }
    } catch (...) {
        if (runtime.device().type() != infinilm::runtime::Device::Type::kCpu) {
            try {
                runtime.syncStream();
            } catch (...) { std::terminate(); }
        }
        throw;
    }
}

infinilm::runtime::Tensor FusedMoE::forward(infinilm::runtime::Runtime &runtime,
                                            const infinilm::runtime::Tensor &hidden_states,
                                            const TopKOutput &topk_output,
                                            const MoeWeights &weights) const {
    runtime.assertOwnerThread();
    if (&runtime != runtime_borrowed_) {
        throw std::invalid_argument("MoE forward Runtime differs from its dispatcher initialization Runtime");
    }
    if (!hidden_states || hidden_states->device() != runtime.device() || !topk_output.topk_ids || topk_output.topk_ids->device() != runtime.device()) {
        throw std::invalid_argument("MoE inputs must belong to Runtime device");
    }
    auto dispatch_output = dispatcher_->dispatch(runtime, hidden_states, topk_output, workspace_);
    auto combine_input = runner_->run(runtime, dispatch_output, weights, workspace_);
    return dispatcher_->combine(runtime, combine_input, workspace_);
}

} // namespace infinilm::layers::moe
