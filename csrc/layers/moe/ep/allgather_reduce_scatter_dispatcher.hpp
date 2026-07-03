#pragma once

#include "base_ep_dispatcher.hpp"

#include <vector>

namespace infinilm::layers::moe {

class AllGatherReduceScatterDispatcher final : public BaseEPDispatcher {
public:
    AllGatherReduceScatterDispatcher(EPConfig ep_config, size_t num_experts);

    DispatchOutput dispatch(const infinicore::Tensor &hidden_states,
                            const TopKOutput &topk_output,
                            MoeWorkspace &workspace) const override;

    infinicore::Tensor combine(const CombineInput &combine_input,
                               MoeWorkspace &workspace) const override;

private:
    void all_gather_dim0_many_(const std::vector<infinicore::Tensor> &outputs,
                               const std::vector<infinicore::Tensor> &inputs) const;
    void reduce_scatter_dim0_(infinicore::Tensor output,
                              const infinicore::Tensor &input) const;
};

} // namespace infinilm::layers::moe
