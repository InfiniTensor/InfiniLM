#include "infinicore/ops/axpy.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Axpy);

Axpy::Axpy(const Tensor &alpha, const Tensor &x, Tensor y) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(alpha, x, y);
    INFINICORE_GRAPH_OP_DISPATCH(y->device().type(), alpha, x, y);
}

void Axpy::execute(const Tensor &alpha, const Tensor &x, Tensor y) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Axpy, alpha, x, y);
}

void axpy_(const Tensor &alpha, const Tensor &x, Tensor y) {
    Axpy::execute(alpha, x, y);
}

} // namespace infinicore::op
