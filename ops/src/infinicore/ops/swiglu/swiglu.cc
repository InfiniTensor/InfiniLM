#include "infinicore/ops/swiglu.hpp"
#include "../../utils.hpp"

namespace infinicore::op {
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(SwiGLU);

SwiGLU::SwiGLU(Tensor c, const Tensor &a, const Tensor &b) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(c, a, b);
    INFINICORE_GRAPH_OP_DISPATCH(c->device().getType(), c, a, b);
}

void SwiGLU::execute(Tensor c, const Tensor &a, const Tensor &b) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(SwiGLU, c, a, b);
}

Tensor swiglu(const Tensor &a, const Tensor &b) {
    auto c = Tensor::empty(a->shape(), a->dtype(), a->device());
    swiglu_(c, a, b);
    return c;
}

void swiglu_(Tensor c, const Tensor &a, const Tensor &b) {
    SwiGLU::execute(c, a, b);
}

} // namespace infinicore::op
