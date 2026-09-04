#include "infinicore/ops/nrm2.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Nrm2);

Nrm2::Nrm2(const Tensor &x, Tensor result) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(x, result);
    INFINICORE_GRAPH_OP_DISPATCH(result->device().type(), x, result);
}

void Nrm2::execute(const Tensor &x, Tensor result) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Nrm2, x, result);
}

Tensor nrm2(const Tensor &x) {
    auto result = Tensor::empty({}, x->dtype(), x->device());
    nrm2_(x, result);
    return result;
}

void nrm2_(const Tensor &x, Tensor result) {
    Nrm2::execute(x, result);
}

} // namespace infinicore::op
