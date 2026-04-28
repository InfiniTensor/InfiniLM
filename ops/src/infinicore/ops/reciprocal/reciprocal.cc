#include "infinicore/ops/reciprocal.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<Reciprocal::schema> &Reciprocal::dispatcher() {
    static common::OpDispatcher<Reciprocal::schema> dispatcher_;
    return dispatcher_;
};

void Reciprocal::execute(Tensor y, Tensor x) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, x);
    infinicore::context::setDevice(y->device());
    dispatcher().lookup(y->device().getType())(y, x);
}

Tensor reciprocal(Tensor x) {
    auto y = Tensor::empty(x->shape(), x->dtype(), x->device());
    reciprocal_(y, x);
    return y;
}

void reciprocal_(Tensor y, Tensor x) {
    Reciprocal::execute(y, x);
}

} // namespace infinicore::op
