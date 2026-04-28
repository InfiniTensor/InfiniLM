#include "infinicore/ops/logaddexp.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<LogAddExp::schema> &LogAddExp::dispatcher() {
    static common::OpDispatcher<LogAddExp::schema> dispatcher_;
    return dispatcher_;
};

void LogAddExp::execute(Tensor c, Tensor a, Tensor b) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(c, a, b);
    infinicore::context::setDevice(c->device());
    dispatcher().lookup(c->device().getType())(c, a, b);
}

Tensor logaddexp(Tensor a, Tensor b) {
    auto c = Tensor::empty(a->shape(), a->dtype(), a->device());
    logaddexp_(c, a, b);
    return c;
}

void logaddexp_(Tensor c, Tensor a, Tensor b) {
    LogAddExp::execute(c, a, b);
}

} // namespace infinicore::op
