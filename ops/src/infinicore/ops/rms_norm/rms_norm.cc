#include "infinicore/ops/rms_norm.hpp"
#include "../../utils.hpp"

namespace infinicore::op {
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(RMSNorm);

RMSNorm::RMSNorm(Tensor y, const Tensor &x, const Tensor &weight, float epsilon) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, x, weight);
    INFINICORE_GRAPH_OP_DISPATCH(y->device().getType(), y, x, weight, epsilon);
}

void RMSNorm::execute(Tensor y, const Tensor &x, const Tensor &weight, float epsilon) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(RMSNorm, y, x, weight, epsilon);
}

Tensor rms_norm(const Tensor &x, const Tensor &weight, float epsilon) {
    auto y = Tensor::empty(x->shape(), x->dtype(), x->device());
    rms_norm_(y, x, weight, epsilon);
    return y;
}

void rms_norm_(Tensor y, const Tensor &x, const Tensor &weight, float epsilon) {
    RMSNorm::execute(y, x, weight, epsilon);
}

} // namespace infinicore::op
