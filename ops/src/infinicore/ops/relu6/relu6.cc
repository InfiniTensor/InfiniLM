#include "infinicore/ops/relu6.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Relu6);

Relu6::Relu6(Tensor out, const Tensor &input) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, input);
    INFINICORE_GRAPH_OP_DISPATCH(out->device().getType(), out, input);
}

void Relu6::execute(Tensor out, const Tensor &input) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Relu6, out, input);
}

Tensor relu6(const Tensor &input) {
    auto out = Tensor::empty(input->shape(), input->dtype(), input->device());
    relu6_(out, input);
    return out;
}

void relu6_(Tensor out, const Tensor &input) {
    Relu6::execute(out, input);
}

} // namespace infinicore::op
