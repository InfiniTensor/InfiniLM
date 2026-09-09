#include "infinicore/ops/rotg.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<Rotg::schema> &Rotg::dispatcher() {
    static common::OpDispatcher<Rotg::schema> dispatcher_;
    return dispatcher_;
};

void Rotg::execute(Tensor x, Tensor y, Tensor c, Tensor s) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(x, y, c, s);
    infinicore::context::setDevice(x->device());
    dispatcher().lookup(x->device().type())(x, y, c, s);
}

void rotg_(Tensor x, Tensor y, Tensor c, Tensor s) {
    Rotg::execute(x, y, c, s);
}

} // namespace infinicore::op
