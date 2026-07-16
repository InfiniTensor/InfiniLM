#pragma once

#include "../device.hpp"
#include "common/op.hpp"

#include <vector>

namespace infinicore::op {

class BlockDiag {
public:
    using schema = void (*)(Tensor, const std::vector<Tensor> &);
    static void execute(Tensor output, const std::vector<Tensor> &inputs);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor block_diag(const std::vector<Tensor> &inputs);
void block_diag_(Tensor output, const std::vector<Tensor> &inputs);

} // namespace infinicore::op
