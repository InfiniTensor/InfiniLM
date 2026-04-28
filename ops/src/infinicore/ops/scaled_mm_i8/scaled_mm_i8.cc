#include "infinicore/ops/scaled_mm_i8.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(I8Gemm);

I8Gemm::I8Gemm(Tensor c, const Tensor &a_p, const Tensor &a_s, const Tensor &b_p, const Tensor &b_s, std::optional<Tensor> bias) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(c, a_p, a_s, b_p, b_s);
    INFINICORE_GRAPH_OP_DISPATCH(c->device().getType(), c, a_p, a_s, b_p, b_s, bias);
}
void I8Gemm::execute(Tensor c, const Tensor &a_p, const Tensor &a_s, const Tensor &b_p, const Tensor &b_s, std::optional<Tensor> bias) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(I8Gemm, c, a_p, a_s, b_p, b_s, bias);
}

void scaled_mm_i8_(Tensor c, const Tensor &a_p, const Tensor &a_s, const Tensor &b_p, const Tensor &b_s, std::optional<Tensor> bias) {
    I8Gemm::execute(c, a_p, a_s, b_p, b_s, bias);
}

} // namespace infinicore::op
