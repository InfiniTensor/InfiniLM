#include "infinicore/ops/logdet.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Logdet);

Logdet::Logdet(Tensor y, const Tensor &x) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, x);
    INFINICORE_GRAPH_OP_DISPATCH(y->device().getType(), y, x);
}

void Logdet::execute(Tensor y, const Tensor &x) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Logdet, y, x);
}

Tensor logdet(const Tensor &x) {
    auto y = Tensor::empty({}, x->dtype(), x->device());
    logdet_(y, x);
    return y;
}

void logdet_(Tensor y, const Tensor &x) {
    Logdet::execute(y, x);
}

} // namespace infinicore::op
