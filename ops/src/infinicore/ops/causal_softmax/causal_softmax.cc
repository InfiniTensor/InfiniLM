#include "infinicore/ops/causal_softmax.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(CausalSoftmax);

CausalSoftmax::CausalSoftmax(Tensor output, const Tensor &input) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);
    INFINICORE_GRAPH_OP_DISPATCH(output->device().getType(), output, input);
}

void CausalSoftmax::execute(Tensor output, const Tensor &input) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(CausalSoftmax, output, input);
}

Tensor causal_softmax(const Tensor &input) {
    auto output = Tensor::empty(input->shape(), input->dtype(), input->device());
    causal_softmax_(output, input);
    return output;
}

void causal_softmax_(Tensor output, const Tensor &input) {
    CausalSoftmax::execute(output, input);
}

} // namespace infinicore::op
