#include "infinicore/ops/blas_dot.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(BlasDot);

BlasDot::BlasDot(const Tensor &x, const Tensor &y, Tensor result) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(x, y, result);
    INFINICORE_GRAPH_OP_DISPATCH(result->device().type(), x, y, result);
}

void BlasDot::execute(const Tensor &x, const Tensor &y, Tensor result) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(BlasDot, x, y, result);
}

Tensor blas_dot(const Tensor &x, const Tensor &y) {
    auto result = Tensor::empty({}, x->dtype(), x->device());
    blas_dot_(x, y, result);
    return result;
}

void blas_dot_(const Tensor &x, const Tensor &y, Tensor result) {
    BlasDot::execute(x, y, result);
}

} // namespace infinicore::op
