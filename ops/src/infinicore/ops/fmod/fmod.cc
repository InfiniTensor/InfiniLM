#include "infinicore/ops/fmod.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<Fmod::schema> &Fmod::dispatcher() {
    static common::OpDispatcher<Fmod::schema> dispatcher_;
    return dispatcher_;
};

void Fmod::execute(Tensor c, Tensor a, Tensor b) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(c, a, b);
    infinicore::context::setDevice(c->device());
    dispatcher().lookup(c->device().getType())(c, a, b);
}

Tensor fmod(Tensor a, Tensor b) {
    auto c = Tensor::empty(a->shape(), a->dtype(), a->device());
    fmod_(c, a, b);
    return c;
}

void fmod_(Tensor c, Tensor a, Tensor b) {
    Fmod::execute(c, a, b);
}

} // namespace infinicore::op
