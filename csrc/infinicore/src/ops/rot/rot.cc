#include "infinicore/ops/rot.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Rot);

Rot::Rot(Tensor x, Tensor y, const Tensor &c, const Tensor &s) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(x, y, c, s);
    INFINICORE_GRAPH_OP_DISPATCH(x->device().type(), x, y, c, s);
}

void Rot::execute(Tensor x, Tensor y, const Tensor &c, const Tensor &s) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Rot, x, y, c, s);
}

void rot_(Tensor x, Tensor y, const Tensor &c, const Tensor &s) {
    Rot::execute(x, y, c, s);
}

} // namespace infinicore::op
