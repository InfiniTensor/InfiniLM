#pragma once

#include "../common/moe_types.hpp"

#include "infinicore/device.hpp"

namespace infinilm::layers::moe {

class BaseDispatcher {
public:
    virtual ~BaseDispatcher() = default;

    virtual void initialize(const infinicore::Device &device,
                            MoeWorkspace &workspace) {
        (void)device;
        (void)workspace;
    }

    virtual DispatchOutput dispatch(const infinicore::Tensor &hidden_states,
                                    const TopKOutput &topk_output,
                                    MoeWorkspace &workspace) const
        = 0;

    virtual infinicore::Tensor combine(const CombineInput &combine_input,
                                       MoeWorkspace &workspace) const
        = 0;
};

} // namespace infinilm::layers::moe
