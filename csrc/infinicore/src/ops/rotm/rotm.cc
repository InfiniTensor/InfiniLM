#include "infinicore/ops/rotm.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<Rotm::schema> &Rotm::dispatcher() {
    static common::OpDispatcher<Rotm::schema> dispatcher_;
    return dispatcher_;
};

void Rotm::execute(Tensor x, Tensor y, Tensor param) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(x, y, param);
    infinicore::context::setDevice(x->device());
    dispatcher().lookup(x->device().type())(x, y, param);
}

void rotm_(Tensor x, Tensor y, Tensor param) {
    Rotm::execute(x, y, param);
}

} // namespace infinicore::op
