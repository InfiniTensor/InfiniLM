#include "infinicore/ops/floor_divide.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<FloorDivide::schema> &FloorDivide::dispatcher() {
    static common::OpDispatcher<FloorDivide::schema> dispatcher_;
    return dispatcher_;
};

void FloorDivide::execute(Tensor c, Tensor a, Tensor b) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(c, a, b);
    infinicore::context::setDevice(c->device());
    dispatcher().lookup(c->device().getType())(c, a, b);
}

Tensor floor_divide(Tensor a, Tensor b) {
    auto c = Tensor::empty(a->shape(), a->dtype(), a->device());
    floor_divide_(c, a, b);
    return c;
}

void floor_divide_(Tensor c, Tensor a, Tensor b) {
    FloorDivide::execute(c, a, b);
}

} // namespace infinicore::op
