#include "infinicore/ops/blas_amax.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(BlasAmax);

BlasAmax::BlasAmax(const Tensor &x, Tensor result) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(x, result);
    INFINICORE_GRAPH_OP_DISPATCH(result->device().type(), x, result);
}

void BlasAmax::execute(const Tensor &x, Tensor result) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(BlasAmax, x, result);
}

Tensor blas_amax(const Tensor &x) {
    auto result = Tensor::empty({}, DataType::kInt32, x->device());
    blas_amax_(x, result);
    return result;
}

void blas_amax_(const Tensor &x, Tensor result) {
    BlasAmax::execute(x, result);
}

} // namespace infinicore::op
