#include "infinicore/ops/bitwise_right_shift.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(BitwiseRightShift);

BitwiseRightShift::BitwiseRightShift(Tensor out, const Tensor &input, const Tensor &other) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, input, other);
    INFINICORE_GRAPH_OP_DISPATCH(out->device().getType(), out, input, other);
}

void BitwiseRightShift::execute(Tensor out, const Tensor &input, const Tensor &other) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(BitwiseRightShift, out, input, other);
}

Tensor bitwise_right_shift(const Tensor &input, const Tensor &other) {
    auto out = Tensor::empty(input->shape(), input->dtype(), input->device());
    bitwise_right_shift_(out, input, other);
    return out;
}

void bitwise_right_shift_(Tensor out, const Tensor &input, const Tensor &other) {
    BitwiseRightShift::execute(out, input, other);
}

} // namespace infinicore::op
