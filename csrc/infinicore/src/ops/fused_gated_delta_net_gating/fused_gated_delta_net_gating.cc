#include "infinicore/ops/fused_gated_delta_net_gating.hpp"

#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(FusedGatedDeltaNetGating);

FusedGatedDeltaNetGating::FusedGatedDeltaNetGating(Tensor g,
                                                   Tensor beta_output,
                                                   const Tensor &A_log,
                                                   const Tensor &a,
                                                   const Tensor &b,
                                                   const Tensor &dt_bias,
                                                   float beta,
                                                   float threshold) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(g, beta_output, A_log, a, b, dt_bias);
    INFINICORE_GRAPH_OP_DISPATCH(g->device().type(), g, beta_output, A_log, a, b, dt_bias, beta, threshold);
}

void FusedGatedDeltaNetGating::execute(Tensor g,
                                       Tensor beta_output,
                                       const Tensor &A_log,
                                       const Tensor &a,
                                       const Tensor &b,
                                       const Tensor &dt_bias,
                                       float beta,
                                       float threshold) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(FusedGatedDeltaNetGating, g, beta_output, A_log, a, b, dt_bias, beta, threshold);
}

static void validate_inputs(const Tensor &A_log,
                            const Tensor &a,
                            const Tensor &b,
                            const Tensor &dt_bias) {
    if (a->shape().size() != 3 || b->shape().size() != 3) {
        throw std::runtime_error("fused_gated_delta_net_gating expects a and b with shape [batch_size, seq_len, hidden]");
    }
    if (a->shape() != b->shape()) {
        throw std::runtime_error("fused_gated_delta_net_gating expects a and b to have the same shape");
    }
    if (A_log->shape().size() != 1 || dt_bias->shape().size() != 1) {
        throw std::runtime_error("fused_gated_delta_net_gating expects A_log and dt_bias with shape [hidden]");
    }
    if (A_log->shape()[0] != a->shape()[2] || dt_bias->shape()[0] != a->shape()[2]) {
        throw std::runtime_error("fused_gated_delta_net_gating hidden dimension mismatch");
    }
}

std::pair<Tensor, Tensor> fused_gated_delta_net_gating(const Tensor &A_log,
                                                       const Tensor &a,
                                                       const Tensor &b,
                                                       const Tensor &dt_bias,
                                                       float beta,
                                                       float threshold) {
    validate_inputs(A_log, a, b, dt_bias);

    Tensor g = Tensor::empty(a->shape(), DataType::kFloat32, a->device());
    Tensor beta_output = Tensor::empty(a->shape(), DataType::kFloat32, a->device());
    fused_gated_delta_net_gating_(g, beta_output, A_log, a, b, dt_bias, beta, threshold);
    return {g, beta_output};
}

void fused_gated_delta_net_gating_(Tensor g,
                                   Tensor beta_output,
                                   const Tensor &A_log,
                                   const Tensor &a,
                                   const Tensor &b,
                                   const Tensor &dt_bias,
                                   float beta,
                                   float threshold) {
    validate_inputs(A_log, a, b, dt_bias);
    if (g->shape() != a->shape() || beta_output->shape() != a->shape()) {
        throw std::runtime_error("fused_gated_delta_net_gating_ expects outputs with shape [batch_size, seq_len, hidden]");
    }
    if (g->dtype() != DataType::kFloat32 || beta_output->dtype() != DataType::kFloat32) {
        throw std::runtime_error("fused_gated_delta_net_gating_ expects float32 outputs");
    }

    FusedGatedDeltaNetGating::execute(g, beta_output, A_log, a, b, dt_bias, beta, threshold);
}

} // namespace infinicore::op
