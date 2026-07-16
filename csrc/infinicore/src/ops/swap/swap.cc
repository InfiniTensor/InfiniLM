#include "infinicore/ops/swap.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Swap);

Swap::Swap(Tensor x, Tensor y) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(x, y);
    INFINICORE_GRAPH_OP_DISPATCH(x->device().type(), x, y);
}

void Swap::execute(Tensor x, Tensor y) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Swap, x, y);
}

void swap_(Tensor x, Tensor y) {
    Swap::execute(x, y);
}

} // namespace infinicore::op
