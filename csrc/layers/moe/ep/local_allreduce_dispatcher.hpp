#pragma once

#include "base_ep_dispatcher.hpp"

namespace infinilm::layers::moe {

class LocalAllReduceDispatcher final : public BaseEPDispatcher {
public:
    LocalAllReduceDispatcher(EPConfig ep_config, size_t num_experts,
                             const engine::distributed::RankInfo &rank_info);

    DispatchOutput dispatch(infinilm::runtime::Runtime &runtime,
                            const infinilm::runtime::Tensor &hidden_states,
                            const TopKOutput &topk_output,
                            MoeWorkspace &workspace) const override;

    infinilm::runtime::Tensor combine(infinilm::runtime::Runtime &runtime,
                               const CombineInput &combine_input,
                               MoeWorkspace &workspace) const override;

};

} // namespace infinilm::layers::moe
