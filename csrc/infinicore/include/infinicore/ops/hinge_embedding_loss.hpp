#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class HingeEmbeddingLoss {
public:
    using schema = void (*)(Tensor, Tensor, Tensor, double, int);
    static void execute(Tensor output, Tensor input, Tensor target, double margin, int reduction);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor hinge_embedding_loss(Tensor input, Tensor target, double margin = 1.0, int reduction = 1);
void hinge_embedding_loss_(Tensor output, Tensor input, Tensor target, double margin, int reduction);

} // namespace infinicore::op
