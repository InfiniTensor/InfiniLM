#include "infinicore/ops/blas_copy.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(BlasCopy);

BlasCopy::BlasCopy(const Tensor &x, Tensor y) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(x, y);
    INFINICORE_GRAPH_OP_DISPATCH(y->device().type(), x, y);
}

void BlasCopy::execute(const Tensor &x, Tensor y) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(BlasCopy, x, y);
}

void blas_copy_(const Tensor &x, Tensor y) {
    BlasCopy::execute(x, y);
}

} // namespace infinicore::op
