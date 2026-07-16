#include "infinicore/ops/sigmoid.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Sigmoid);

Sigmoid::Sigmoid(Tensor output, const Tensor &input) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);
    INFINICORE_GRAPH_OP_DISPATCH(output->device().type(), output, input);
}

void Sigmoid::execute(Tensor output, const Tensor &input) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Sigmoid, output, input);
}

Tensor sigmoid(const Tensor &input) {
    auto output = Tensor::empty(input->shape(), input->dtype(), input->device());
    sigmoid_(output, input);
    return output;
}

void sigmoid_(Tensor output, const Tensor &input) {
    Sigmoid::execute(output, input);
}

} // namespace infinicore::op
