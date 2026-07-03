#pragma once

#include "base_ep_dispatcher.hpp"

namespace infinilm::layers::moe {

class LocalAllReduceDispatcher final : public BaseEPDispatcher {
public:
    LocalAllReduceDispatcher(EPConfig ep_config, size_t num_experts);

    DispatchOutput dispatch(const infinicore::Tensor &hidden_states,
                            const TopKOutput &topk_output,
                            MoeWorkspace &workspace) const override;

    infinicore::Tensor combine(const CombineInput &combine_input,
                               MoeWorkspace &workspace) const override;

private:
    void allreduce_(infinicore::Tensor tensor) const;
};

} // namespace infinilm::layers::moe
