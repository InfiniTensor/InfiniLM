#include "infinicore/ops/broadcast_to.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<BroadcastTo::schema> &BroadcastTo::dispatcher() {
    static common::OpDispatcher<BroadcastTo::schema> dispatcher_;
    return dispatcher_;
};

void BroadcastTo::execute(Tensor y, Tensor x) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, x);
    infinicore::context::setDevice(y->device());
    dispatcher().lookup(y->device().getType())(y, x);
}

Tensor broadcast_to(Tensor x, const std::vector<int64_t> &shape) {
    Shape target_shape(shape.begin(), shape.end());

    auto y = Tensor::empty(target_shape, x->dtype(), x->device());
    broadcast_to_(y, x);
    return y;
}

void broadcast_to_(Tensor y, Tensor x) {
    BroadcastTo::execute(y, x);
}

} // namespace infinicore::op
