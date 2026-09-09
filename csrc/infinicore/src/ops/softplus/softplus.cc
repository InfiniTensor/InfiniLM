#include "infinicore/ops/softplus.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<Softplus::schema> &Softplus::dispatcher() {
    static common::OpDispatcher<Softplus::schema> dispatcher_;
    return dispatcher_;
};

void Softplus::execute(Tensor y, Tensor x, float beta, float threshold) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, x);
    infinicore::context::setDevice(y->device());

    dispatcher().lookup(y->device().type())(y, x, beta, threshold);
}

Tensor softplus(Tensor x, float beta, float threshold) {
    auto y = Tensor::empty(x->shape(), x->dtype(), x->device());
    softplus_(y, x, beta, threshold);
    return y;
}

void softplus_(Tensor y, Tensor x, float beta, float threshold) {
    Softplus::execute(y, x, beta, threshold);
}

} // namespace infinicore::op
