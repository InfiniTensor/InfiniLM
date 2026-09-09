#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class TripletMarginLoss {
public:
    // Schema signature: output, anchor, positive, negative, margin, p, eps, swap, reduction
    using schema = void (*)(Tensor, Tensor, Tensor, Tensor, float, int64_t, float, bool, int64_t);

    static void execute(Tensor output, Tensor anchor, Tensor positive, Tensor negative, float margin, int64_t p, float eps, bool swap, int64_t reduction);
    static common::OpDispatcher<schema> &dispatcher();
};

// Functional API
// reduction: 0=None, 1=Mean, 2=Sum
Tensor triplet_margin_loss(Tensor anchor, Tensor positive, Tensor negative, float margin = 1.0f, int64_t p = 2, float eps = 1e-6f, bool swap = false, int64_t reduction = 1);

// In-place / Explicit Output API
void triplet_margin_loss_(Tensor output, Tensor anchor, Tensor positive, Tensor negative, float margin, int64_t p, float eps, bool swap, int64_t reduction);

} // namespace infinicore::op
