#include "infinicore/ops/asinh.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<Asinh::schema> &Asinh::dispatcher() {
    static common::OpDispatcher<Asinh::schema> dispatcher_;
    return dispatcher_;
};

void Asinh::execute(Tensor y, Tensor x) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, x);
    infinicore::context::setDevice(y->device());
    dispatcher().lookup(y->device().getType())(y, x);
}

Tensor asinh(Tensor x) {
    auto y = Tensor::empty(x->shape(), x->dtype(), x->device());
    asinh_(y, x);
    return y;
}

void asinh_(Tensor y, Tensor x) {
    Asinh::execute(y, x);
}

} // namespace infinicore::op
