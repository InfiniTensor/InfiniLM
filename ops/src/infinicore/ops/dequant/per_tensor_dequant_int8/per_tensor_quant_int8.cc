#include "../../../utils.hpp"
#include "infinicore/ops/per_tensor_dequant_i8.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(PerTensorDequantI8);

PerTensorDequantI8::PerTensorDequantI8(Tensor x, const Tensor &x_packed, const Tensor &x_scale, const Tensor &x_zero) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(x, x_packed, x_scale, x_zero);
    INFINICORE_GRAPH_OP_DISPATCH(x->device().getType(), x, x_packed, x_scale, x_zero);
}

void PerTensorDequantI8::execute(Tensor x, const Tensor &x_packed, const Tensor &x_scale, const Tensor &x_zero) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(PerTensorDequantI8, x, x_packed, x_scale, x_zero);
}

void per_tensor_dequant_i8_(Tensor x, const Tensor &x_packed, const Tensor &x_scale, const Tensor &x_zero) {
    PerTensorDequantI8::execute(x, x_packed, x_scale, x_zero);
}
} // namespace infinicore::op
