#include "infinicore/ops/rotmg.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<Rotmg::schema> &Rotmg::dispatcher() {
    static common::OpDispatcher<Rotmg::schema> dispatcher_;
    return dispatcher_;
};

void Rotmg::execute(Tensor d1, Tensor d2, Tensor x1, Tensor y1, Tensor param) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(d1, d2, x1, y1, param);
    infinicore::context::setDevice(d1->device());
    dispatcher().lookup(d1->device().type())(d1, d2, x1, y1, param);
}

void rotmg_(Tensor d1, Tensor d2, Tensor x1, Tensor y1, Tensor param) {
    Rotmg::execute(d1, d2, x1, y1, param);
}

} // namespace infinicore::op
