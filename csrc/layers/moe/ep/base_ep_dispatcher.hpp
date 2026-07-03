#pragma once

#include "../dispatcher/base_dispatcher.hpp"
#include "ep_config.hpp"

#include <infiniccl.h>
#include <vector>

namespace infinilm::layers::moe {

class BaseEPDispatcher : public BaseDispatcher {
public:
    BaseEPDispatcher(EPConfig ep_config, size_t num_experts);

    void initialize(const infinicore::Device &device,
                    MoeWorkspace &workspace) override;

protected:
    std::vector<size_t> equal_split_sizes(size_t local_dim0) const;
    infinicore::Tensor expert_map(const infinicore::Device &device) const;

    EPConfig config_;
    size_t num_experts_ = 0;
    infinicclComm_t communicator_ = nullptr;

private:
    mutable infinicore::Tensor expert_map_;
};

} // namespace infinilm::layers::moe
