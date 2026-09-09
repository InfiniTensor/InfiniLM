#include "infinicore/ops/scal.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Scal);

Scal::Scal(const Tensor &alpha, Tensor x) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(alpha, x);
    INFINICORE_GRAPH_OP_DISPATCH(x->device().type(), alpha, x);
}

void Scal::execute(const Tensor &alpha, Tensor x) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Scal, alpha, x);
}

void scal_(const Tensor &alpha, Tensor x) {
    Scal::execute(alpha, x);
}

} // namespace infinicore::op
