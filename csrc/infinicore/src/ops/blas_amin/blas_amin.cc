#include "infinicore/ops/blas_amin.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(BlasAmin);

BlasAmin::BlasAmin(const Tensor &x, Tensor result) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(x, result);
    INFINICORE_GRAPH_OP_DISPATCH(result->device().type(), x, result);
}

void BlasAmin::execute(const Tensor &x, Tensor result) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(BlasAmin, x, result);
}

Tensor blas_amin(const Tensor &x) {
    auto result = Tensor::empty({}, DataType::kInt32, x->device());
    blas_amin_(x, result);
    return result;
}

void blas_amin_(const Tensor &x, Tensor result) {
    BlasAmin::execute(x, result);
}

} // namespace infinicore::op
