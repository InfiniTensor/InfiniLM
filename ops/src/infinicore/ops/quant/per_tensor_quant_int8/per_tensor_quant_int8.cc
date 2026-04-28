#include "../../../utils.hpp"
#include "infinicore/ops/per_tensor_quant_i8.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(PerTensorQuantI8);

PerTensorQuantI8::PerTensorQuantI8(const Tensor &x, Tensor x_packed, Tensor x_scale, Tensor x_zero, bool is_static) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(x, x_packed, x_scale, x_zero);
    INFINICORE_GRAPH_OP_DISPATCH(x->device().getType(), x, x_packed, x_scale, x_zero, is_static);
}

void PerTensorQuantI8::execute(const Tensor &x, Tensor x_packed, Tensor x_scale, Tensor x_zero, bool is_static) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(PerTensorQuantI8, x, x_packed, x_scale, x_zero, is_static);
}

void per_tensor_quant_i8_(const Tensor &x, Tensor x_packed, Tensor x_scale, Tensor x_zero, bool is_static) {
    PerTensorQuantI8::execute(x, x_packed, x_scale, x_zero, is_static);
}

Tensor per_tensor_quant_i8(const Tensor &x, Tensor x_scale, Tensor x_zero, bool is_static) {
    auto x_packed = Tensor::strided_empty(x->shape(), x->strides(), infinicore::DataType::I8, x->device());
    PerTensorQuantI8::execute(x, x_packed, x_scale, x_zero, is_static);
    return x_packed;
}
} // namespace infinicore::op
