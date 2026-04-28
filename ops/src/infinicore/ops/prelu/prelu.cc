#include "infinicore/ops/prelu.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Prelu);

Prelu::Prelu(Tensor out, const Tensor &input, const Tensor &weight) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, input, weight);
    INFINICORE_GRAPH_OP_DISPATCH(out->device().getType(), out, input, weight);
}

void Prelu::execute(Tensor out, const Tensor &input, const Tensor &weight) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Prelu, out, input, weight);
}

Tensor prelu(const Tensor &input, const Tensor &weight) {
    auto out = Tensor::empty(input->shape(), input->dtype(), input->device());
    prelu_(out, input, weight);
    return out;
}

void prelu_(Tensor out, const Tensor &input, const Tensor &weight) {
    Prelu::execute(out, input, weight);
}

} // namespace infinicore::op
