#include "infinicore/ops/add.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Add);

Add::Add(Tensor c, const Tensor &a, const Tensor &b) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(c, a, b);
    INFINICORE_GRAPH_OP_DISPATCH(c->device().getType(), c, a, b);
}

void Add::execute(Tensor c, const Tensor &a, const Tensor &b) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Add, c, a, b);
}

Tensor add(const Tensor &a, const Tensor &b) {
    auto c = Tensor::empty(a->shape(), a->dtype(), a->device());
    add_(c, a, b);
    return c;
}

void add_(Tensor c, const Tensor &a, const Tensor &b) {
    Add::execute(c, a, b);
}

} // namespace infinicore::op
