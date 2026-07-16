#include "infinicore/ops/asum.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Asum);

Asum::Asum(const Tensor &x, Tensor result) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(x, result);
    INFINICORE_GRAPH_OP_DISPATCH(result->device().type(), x, result);
}

void Asum::execute(const Tensor &x, Tensor result) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Asum, x, result);
}

Tensor asum(const Tensor &x) {
    auto result = Tensor::empty({}, x->dtype(), x->device());
    asum_(x, result);
    return result;
}

void asum_(const Tensor &x, Tensor result) {
    Asum::execute(x, result);
}

} // namespace infinicore::op
