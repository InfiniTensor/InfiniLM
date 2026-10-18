#pragma once

#include "../../../engine/distributed/communication_group.hpp"
#include "../dispatcher/base_dispatcher.hpp"
#include "ep_config.hpp"

#include <infiniccl/infiniccl.h>
#include <memory>
#include <vector>

namespace infinilm::layers::moe {

class BaseEPDispatcher : public BaseDispatcher {
public:
    BaseEPDispatcher(EPConfig ep_config, size_t num_experts,
                     const engine::distributed::RankInfo &rank_info);

    void initialize(infinilm::runtime::Runtime &runtime,
                    MoeWorkspace &workspace) override;

protected:
    void assertInitializedRuntime(infinilm::runtime::Runtime &runtime) const;
    std::vector<size_t> equal_split_sizes(size_t local_dim0) const;
    infinilm::runtime::Tensor expert_map(infinilm::runtime::Runtime &runtime) const;

    EPConfig config_;
    size_t num_experts_ = 0;
    infinicclComm_t communicator_ = nullptr;
    infinilm::runtime::Device device_;

private:
    std::shared_ptr<infinilm::runtime::Runtime> runtime_owner_;
    mutable infinilm::runtime::Tensor expert_map_;
};

} // namespace infinilm::layers::moe
