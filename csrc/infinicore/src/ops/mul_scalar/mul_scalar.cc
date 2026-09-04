#include "infinicore/ops/mul_scalar.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(MulScalar);

MulScalar::MulScalar(Tensor c, const Tensor &a, double alpha) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(c, a);
    INFINICORE_GRAPH_OP_DISPATCH(c->device().type(), c, a, alpha);
}

void MulScalar::execute(Tensor c, const Tensor &a, double alpha) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(MulScalar, c, a, alpha);
}

Tensor mul_scalar(const Tensor &a, double alpha) {
    auto c = Tensor::empty(a->shape(), a->dtype(), a->device());
    mul_scalar_(c, a, alpha);
    return c;
}

void mul_scalar_(Tensor c, const Tensor &a, double alpha) {
    MulScalar::execute(c, a, alpha);
}

} // namespace infinicore::op
