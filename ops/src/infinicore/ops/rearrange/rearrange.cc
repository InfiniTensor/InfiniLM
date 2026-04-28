#include "infinicore/ops/rearrange.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Rearrange);

Rearrange::Rearrange(Tensor y, const Tensor &x) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, x);
    INFINICORE_GRAPH_OP_DISPATCH(y->device().getType(), y, x);
}

void Rearrange::execute(Tensor y, const Tensor &x) {
    auto op = std::make_shared<Rearrange>(y, x);
    if (context::isGraphRecording()) {
        context::addGraphOperator(op);
    } else {
        op->run();
    }
}

Tensor rearrange(const Tensor &x) {
    auto y = Tensor::empty(x->shape(), x->dtype(), x->device());
    rearrange_(y, x);
    return y;
}

void rearrange_(Tensor y, const Tensor &x) {
    Rearrange::execute(y, x);
}

} // namespace infinicore::op
