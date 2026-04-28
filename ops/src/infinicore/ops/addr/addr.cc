#include "infinicore/ops/addr.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<Addr::schema> &Addr::dispatcher() {
    static common::OpDispatcher<Addr::schema> dispatcher_;
    return dispatcher_;
};

void Addr::execute(Tensor out, Tensor input, Tensor vec1, Tensor vec2, float beta, float alpha) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, input, vec1, vec2);
    infinicore::context::setDevice(out->device());
    dispatcher().lookup(out->device().getType())(out, input, vec1, vec2, beta, alpha);
}

Tensor addr(Tensor input, Tensor vec1, Tensor vec2, float beta, float alpha) {

    size_t n = vec1->shape()[0];
    size_t m = vec2->shape()[0];

    // Create output tensor
    Tensor out = Tensor::empty({n, m}, input->dtype(), input->device());
    addr_(out, input, vec1, vec2, beta, alpha);
    return out;
}

void addr_(Tensor out, Tensor input, Tensor vec1, Tensor vec2, float beta, float alpha) {
    Addr::execute(out, input, vec1, vec2, beta, alpha);
}

} // namespace infinicore::op
