#include "infinicore/ops/add_rms_norm.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(AddRMSNorm);

AddRMSNorm::AddRMSNorm(Tensor y, Tensor residual_out, const Tensor &a, const Tensor &b, const Tensor &weight, float epsilon) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, residual_out, a, b, weight);
    INFINICORE_GRAPH_OP_DISPATCH(y->device().getType(), y, residual_out, a, b, weight, epsilon);
}

void AddRMSNorm::execute(Tensor y, Tensor residual_out, const Tensor &a, const Tensor &b, const Tensor &weight, float epsilon) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(AddRMSNorm, y, residual_out, a, b, weight, epsilon);
}

std::pair<Tensor, Tensor> add_rms_norm(const Tensor &a, const Tensor &b, const Tensor &weight, float epsilon) {
    auto y = Tensor::empty(a->shape(), a->dtype(), a->device());
    auto residual_out = Tensor::empty(a->shape(), a->dtype(), a->device());
    add_rms_norm_(y, residual_out, a, b, weight, epsilon);
    return std::make_pair(y, residual_out);
}

void add_rms_norm_(Tensor out, Tensor residual, const Tensor &a, const Tensor &b, const Tensor &weight, float epsilon) {
    AddRMSNorm::execute(out, residual, a, b, weight, epsilon);
}

void add_rms_norm_inplace(Tensor input, Tensor residual, const Tensor &weight, float epsilon) {
    add_rms_norm_(input, residual, input, residual, weight, epsilon);
}

} // namespace infinicore::op
