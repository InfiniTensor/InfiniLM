#include "infinicore/ops/dist.hpp"

#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Dist);

Dist::Dist(Tensor y, const Tensor &x1, const Tensor &x2, double p) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, x1, x2);
    INFINICORE_GRAPH_OP_DISPATCH(y->device().getType(), y, x1, x2, p);
}

void Dist::execute(Tensor y, const Tensor &x1, const Tensor &x2, double p) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Dist, y, x1, x2, p);
}

Tensor dist(const Tensor &x1, const Tensor &x2, double p) {
    auto y = Tensor::empty({}, x1->dtype(), x1->device());
    dist_(y, x1, x2, p);
    return y;
}

void dist_(Tensor y, const Tensor &x1, const Tensor &x2, double p) {
    Dist::execute(y, x1, x2, p);
}

} // namespace infinicore::op
