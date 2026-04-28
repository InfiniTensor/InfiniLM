#include "infinicore/ops/layer_norm.hpp"
#include "../../utils.hpp"

namespace infinicore::op {
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(LayerNorm);

LayerNorm::LayerNorm(Tensor y, Tensor standardization, Tensor std_deviation, const Tensor &x, const Tensor &weight, const Tensor &bias, float epsilon) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, standardization, std_deviation, x, weight);
    INFINICORE_GRAPH_OP_DISPATCH(y->device().getType(), y, standardization, std_deviation, x, weight, bias, epsilon);
}

void LayerNorm::execute(Tensor y, Tensor standardization, Tensor std_deviation, const Tensor &x, const Tensor &weight, const Tensor &bias, float epsilon) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(LayerNorm, y, standardization, std_deviation, x, weight, bias, epsilon);
}

Tensor layer_norm(const Tensor &x, const Tensor &weight, const Tensor &bias, float epsilon) {
    auto y = Tensor::empty(x->shape(), x->dtype(), x->device());
    auto reduced_shape = x->shape();
    reduced_shape.pop_back();
    auto standardization = Tensor::empty(x->shape(), x->dtype(), x->device());
    auto std_deviation = Tensor::empty(reduced_shape, x->dtype(), x->device());
    layer_norm_(y, standardization, std_deviation, x, weight, bias, epsilon);
    return y;
}

void layer_norm_(Tensor y, Tensor standardization, Tensor std_deviation, const Tensor &x, const Tensor &weight, const Tensor &bias, float epsilon) {
    LayerNorm::execute(y, standardization, std_deviation, x, weight, bias, epsilon);
}

void layer_norm_(Tensor y, const Tensor &x, const Tensor &weight, const Tensor &bias, float epsilon) {
    auto reduced_shape = x->shape();
    reduced_shape.pop_back();
    auto standardization = Tensor::empty(x->shape(), x->dtype(), x->device());
    auto std_deviation = Tensor::empty(reduced_shape, x->dtype(), x->device());
    LayerNorm::execute(y, standardization, std_deviation, x, weight, bias, epsilon);
}

void layer_norm_for_pybind(Tensor y, const Tensor &x, const Tensor &weight, const Tensor &bias, float epsilon) {
    layer_norm_(y, x, weight, bias, epsilon);
}

} // namespace infinicore::op
