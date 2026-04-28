#include "infinicore/ops/mul.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Mul);

Mul::Mul(Tensor c, const Tensor &a, const Tensor &b) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(c, a, b);
    INFINICORE_GRAPH_OP_DISPATCH(c->device().getType(), c, a, b);
}

void Mul::execute(Tensor c, const Tensor &a, const Tensor &b) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Mul, c, a, b);
}

Tensor mul(const Tensor &a, const Tensor &b) {
    auto c = Tensor::empty(a->shape(), a->dtype(), a->device());
    mul_(c, a, b);
    return c;
}

void mul_(Tensor c, const Tensor &a, const Tensor &b) {
    Mul::execute(c, a, b);
}

} // namespace infinicore::op
