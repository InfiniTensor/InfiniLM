#include "infinicore/ops/digamma.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Digamma);

Digamma::Digamma(Tensor y, const Tensor &x) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, x);
    INFINICORE_GRAPH_OP_DISPATCH(y->device().getType(), y, x);
}

void Digamma::execute(Tensor y, const Tensor &x) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Digamma, y, x);
}

Tensor digamma(const Tensor &x) {
    auto y = Tensor::empty(x->shape(), x->dtype(), x->device());
    digamma_(y, x);
    return y;
}

void digamma_(Tensor y, const Tensor &x) {
    Digamma::execute(y, x);
}

} // namespace infinicore::op
