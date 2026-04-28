#include "infinicore/ops/dequantize_awq.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DequantizeAWQ);

DequantizeAWQ::DequantizeAWQ(Tensor x, const Tensor &x_packed, const Tensor &x_scale, const Tensor &x_zeros) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(x, x_packed, x_scale, x_zeros);
    INFINICORE_GRAPH_OP_DISPATCH(x->device().getType(), x, x_packed, x_scale, x_zeros);
}

void DequantizeAWQ::execute(Tensor x, const Tensor &x_packed, const Tensor &x_scale, const Tensor &x_zeros) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(DequantizeAWQ, x, x_packed, x_scale, x_zeros);
}

void dequantize_awq_(Tensor x, const Tensor &x_packed, const Tensor &x_scale, const Tensor &x_zeros) {
    DequantizeAWQ::execute(x, x_packed, x_scale, x_zeros);
}
} // namespace infinicore::op
