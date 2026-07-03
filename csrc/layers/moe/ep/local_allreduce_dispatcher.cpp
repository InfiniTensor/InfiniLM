#include "local_allreduce_dispatcher.hpp"

#include "infinicore/ops/distributed/allreduce.hpp"

#include <utility>

namespace infinilm::layers::moe {

LocalAllReduceDispatcher::LocalAllReduceDispatcher(EPConfig ep_config,
                                                   size_t num_experts)
    : BaseEPDispatcher(std::move(ep_config), num_experts) {}

void LocalAllReduceDispatcher::allreduce_(infinicore::Tensor tensor) const {
    if (!tensor) {
        return;
    }
    infinicore::op::distributed::allreduce_(tensor, tensor, INFINICCL_SUM, communicator_);
}

DispatchOutput LocalAllReduceDispatcher::dispatch(
    const infinicore::Tensor &hidden_states,
    const TopKOutput &topk_output,
    MoeWorkspace &workspace) const {
    (void)workspace;
    if (config_.ep_size == 1) {
        return DispatchOutput{
            DispatchOutputFormat::Standard,
            hidden_states,
            infinicore::Tensor(),
            topk_output,
            infinicore::Tensor(),
        };
    }
    return DispatchOutput{
        DispatchOutputFormat::Standard,
        hidden_states,
        infinicore::Tensor(),
        topk_output,
        expert_map(hidden_states->device()),
    };
}

infinicore::Tensor LocalAllReduceDispatcher::combine(
    const CombineInput &combine_input,
    MoeWorkspace &workspace) const {
    (void)workspace;
    if (config_.ep_size == 1) {
        return combine_input.hidden_states;
    }
    allreduce_(combine_input.hidden_states);
    return combine_input.hidden_states;
}

} // namespace infinilm::layers::moe
