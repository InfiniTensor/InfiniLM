#pragma once

#include "../common/moe_types.hpp"

#include "runtime/device.hpp"

namespace infinilm::runtime {
class Runtime;
}

namespace infinilm::layers::moe {

class BaseDispatcher {
public:
    virtual ~BaseDispatcher() = default;

    virtual void initialize(infinilm::runtime::Runtime &runtime,
                            MoeWorkspace &workspace) {
        (void)runtime;
        (void)workspace;
    }

    virtual DispatchOutput dispatch(infinilm::runtime::Runtime &runtime,
                                    const infinilm::runtime::Tensor &hidden_states,
                                    const TopKOutput &topk_output,
                                    MoeWorkspace &workspace) const
        = 0;

    virtual infinilm::runtime::Tensor combine(infinilm::runtime::Runtime &runtime,
                                              const CombineInput &combine_input,
                                              MoeWorkspace &workspace) const
        = 0;
};

} // namespace infinilm::layers::moe
