#include "local_allreduce_dispatcher.hpp"

#include "../../../engine/distributed/communication_group.hpp"
#include "runtime/runtime.hpp"

#include <stdexcept>
#include <utility>

namespace infinilm::layers::moe {

LocalAllReduceDispatcher::LocalAllReduceDispatcher(EPConfig ep_config,
                                                   size_t num_experts,
                                                   const engine::distributed::RankInfo &rank_info)
    : BaseEPDispatcher(std::move(ep_config), num_experts, rank_info) {}

DispatchOutput LocalAllReduceDispatcher::dispatch(
    infinilm::runtime::Runtime &runtime,
    const infinilm::runtime::Tensor &hidden_states,
    const TopKOutput &topk_output,
    MoeWorkspace &workspace) const {
    assertInitializedRuntime(runtime);
    if (hidden_states && runtime.device() != hidden_states->device()) {
        throw std::invalid_argument("MoE local-allreduce input device differs from Runtime device");
    }
    if ((topk_output.topk_weights && runtime.device() != topk_output.topk_weights->device()) || (topk_output.topk_ids && runtime.device() != topk_output.topk_ids->device())) {
        throw std::invalid_argument("MoE local-allreduce routing device differs from Runtime device");
    }
    (void)workspace;
    if (config_.ep_size == 1) {
        return DispatchOutput{
            DispatchOutputFormat::Standard,
            hidden_states,
            infinilm::runtime::Tensor(),
            topk_output,
            infinilm::runtime::Tensor(),
        };
    }
    return DispatchOutput{
        DispatchOutputFormat::Standard,
        hidden_states,
        infinilm::runtime::Tensor(),
        topk_output,
        expert_map(runtime),
    };
}

infinilm::runtime::Tensor LocalAllReduceDispatcher::combine(
    infinilm::runtime::Runtime &runtime,
    const CombineInput &combine_input,
    MoeWorkspace &workspace) const {
    assertInitializedRuntime(runtime);
    if (combine_input.hidden_states && runtime.device() != combine_input.hidden_states->device()) {
        throw std::invalid_argument("MoE local-allreduce combine device differs from Runtime device");
    }
    (void)workspace;
    if (config_.ep_size == 1) {
        return combine_input.hidden_states;
    }
    if (combine_input.hidden_states) {
        infinilm::engine::distributed::allreduce_(runtime, combine_input.hidden_states,
                                                  combine_input.hidden_states,
                                                  infinicclSum, communicator_);
    }
    return combine_input.hidden_states;
}

} // namespace infinilm::layers::moe
