#include "../../utils.hpp"
#include "infinicore/ops/logaddexp2.hpp"

namespace infinicore::op {

common::OpDispatcher<LogAddExp2::schema> &LogAddExp2::dispatcher() {
    static common::OpDispatcher<LogAddExp2::schema> dispatcher_;
    return dispatcher_;
};

void LogAddExp2::execute(Tensor c, Tensor a, Tensor b) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(c, a, b);
    infinicore::context::setDevice(c->device());
    dispatcher().lookup(c->device().getType())(c, a, b);
}

Tensor logaddexp2(Tensor a, Tensor b) {
    auto c = Tensor::empty(a->shape(), a->dtype(), a->device());
    logaddexp2_(c, a, b);
    return c;
}

void logaddexp2_(Tensor c, Tensor a, Tensor b) {
    LogAddExp2::execute(c, a, b);
}

} // namespace infinicore::op
