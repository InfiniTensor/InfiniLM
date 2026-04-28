#include "infinicore/ops/equal.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<Equal::schema> &Equal::dispatcher() {
    static common::OpDispatcher<Equal::schema> dispatcher_;
    return dispatcher_;
};

void Equal::execute(Tensor out, Tensor a, Tensor b) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, a, b);
    infinicore::context::setDevice(out->device());
    dispatcher().lookup(out->device().getType())(out, a, b);
}

Tensor equal(Tensor a, Tensor b) {
    auto out = Tensor::empty(a->shape(), DataType::BOOL, a->device());
    equal_(out, a, b);
    return out;
}

void equal_(Tensor out, Tensor a, Tensor b) {
    if (out->dtype() != DataType::BOOL) {
        throw std::runtime_error("Equal expects bool output tensor.");
    }
    Equal::execute(out, a, b);
}

} // namespace infinicore::op
