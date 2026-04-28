#include "infinicore/ops/fmin.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<Fmin::schema> &Fmin::dispatcher() {
    static common::OpDispatcher<Fmin::schema> dispatcher_;
    return dispatcher_;
}

void Fmin::execute(Tensor c, Tensor a, Tensor b) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(c, a, b);
    infinicore::context::setDevice(c->device());
    dispatcher().lookup(c->device().getType())(c, a, b);
}

Tensor fmin(Tensor a, Tensor b) {
    auto c = Tensor::empty(a->shape(), a->dtype(), a->device());
    fmin_(c, a, b);
    return c;
}

void fmin_(Tensor c, Tensor a, Tensor b) {
    Fmin::execute(c, a, b);
}

} // namespace infinicore::op
