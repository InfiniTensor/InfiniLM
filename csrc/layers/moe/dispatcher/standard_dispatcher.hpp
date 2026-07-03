#pragma once

#include "../common/moe_types.hpp"
#include "base_dispatcher.hpp"

#include <infiniccl.h>

namespace infinilm::layers::moe {

class StandardDispatcher final : public BaseDispatcher {
public:
    StandardDispatcher();

    DispatchOutput dispatch(const infinicore::Tensor &hidden_states,
                            const TopKOutput &topk_output,
                            MoeWorkspace &workspace) const override;

    infinicore::Tensor combine(const CombineInput &combine_input,
                               MoeWorkspace &workspace) const override;

private:
    size_t tp_size_ = 1;
    infinicclComm_t communicator_ = nullptr;
};

} // namespace infinilm::layers::moe
