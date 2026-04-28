#include "infinicore/ops/softsign.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<Softsign::schema> &Softsign::dispatcher() {
    static common::OpDispatcher<Softsign::schema> dispatcher_;
    return dispatcher_;
};

void Softsign::execute(Tensor y, Tensor x) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, x);
    infinicore::context::setDevice(y->device());
    dispatcher().lookup(y->device().getType())(y, x);
}

Tensor softsign(Tensor x) {
    auto y = Tensor::empty(x->shape(), x->dtype(), x->device());
    softsign_(y, x);
    return y;
}

void softsign_(Tensor y, Tensor x) {
    Softsign::execute(y, x);
}

} // namespace infinicore::op
