#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class TripletMarginWithDistanceLoss {
public:
    // Schema signature: output(out), anchor, positive, negative, margin, swap, reduction
    using schema = void (*)(Tensor, Tensor, Tensor, Tensor, double, bool, int64_t);

    static void execute(Tensor output, Tensor anchor, Tensor positive, Tensor negative, double margin, bool swap, int64_t reduction);
    static common::OpDispatcher<schema> &dispatcher();
};

// Functional API: Returns the result tensor
// margin default 1.0, swap default false, reduction default 1 (Mean) typically
Tensor triplet_margin_with_distance_loss(Tensor anchor, Tensor positive, Tensor negative, double margin = 1.0, bool swap = false, int64_t reduction = 1);

// In-place/Output-provided API
void triplet_margin_with_distance_loss_(Tensor output, Tensor anchor, Tensor positive, Tensor negative, double margin, bool swap, int64_t reduction);

} // namespace infinicore::op
