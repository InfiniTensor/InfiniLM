#pragma once

#include "../common/moe_types.hpp"

namespace infinilm::layers::moe {

class MoeRunnerCore {
public:
    virtual ~MoeRunnerCore() = default;

    virtual CombineInput run(const DispatchOutput &dispatch_output,
                             const MoeWeights &weights,
                             MoeWorkspace &workspace) const
        = 0;
};

} // namespace infinilm::layers::moe
