#pragma once

#include "../../../engine/distributed/communication_group.hpp"
#include "../common/moe_types.hpp"
#include "base_dispatcher.hpp"
#include <memory>

#include <infiniccl/infiniccl.h>

namespace infinilm::layers::moe {

class StandardDispatcher final : public BaseDispatcher {
public:
    explicit StandardDispatcher(const engine::distributed::RankInfo &rank_info);
    void initialize(infinilm::runtime::Runtime &runtime, MoeWorkspace &workspace) override;

    DispatchOutput dispatch(infinilm::runtime::Runtime &runtime,
                            const infinilm::runtime::Tensor &hidden_states,
                            const TopKOutput &topk_output,
                            MoeWorkspace &workspace) const override;

    infinilm::runtime::Tensor combine(infinilm::runtime::Runtime &runtime,
                                      const CombineInput &combine_input,
                                      MoeWorkspace &workspace) const override;

private:
    void assertInitializedRuntime(infinilm::runtime::Runtime &runtime) const;
    size_t tp_size_ = 1;
    infinicclComm_t communicator_ = nullptr;
    infinilm::runtime::Device device_;
    std::shared_ptr<infinilm::runtime::Runtime> runtime_owner_;
};

} // namespace infinilm::layers::moe
