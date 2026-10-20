#pragma once

#include "base_ep_dispatcher.hpp"

#include <vector>

namespace infinilm::layers::moe {

class AllGatherReduceScatterDispatcher final : public BaseEPDispatcher {
public:
    AllGatherReduceScatterDispatcher(EPConfig ep_config, size_t num_experts,
                                     const engine::distributed::RankInfo &rank_info);

    DispatchOutput dispatch(infinilm::runtime::Runtime &runtime,
                            const infinilm::runtime::Tensor &hidden_states,
                            const TopKOutput &topk_output,
                            MoeWorkspace &workspace) const override;

    infinilm::runtime::Tensor combine(infinilm::runtime::Runtime &runtime,
                                      const CombineInput &combine_input,
                                      MoeWorkspace &workspace) const override;

private:
    void all_gather_dim0_many_(infinilm::runtime::Runtime &runtime,
                               const std::vector<infinilm::runtime::Tensor> &outputs,
                               const std::vector<infinilm::runtime::Tensor> &inputs) const;
    void reduce_scatter_dim0_(infinilm::runtime::Runtime &runtime,
                              infinilm::runtime::Tensor output,
                              const infinilm::runtime::Tensor &input) const;
};

} // namespace infinilm::layers::moe
