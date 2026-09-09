#include "infinicore/ops/causal_conv1d.hpp"
#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(CausalConv1d);

CausalConv1d::CausalConv1d(Tensor out,
                           Tensor conv_state,
                           std::optional<Tensor> final_conv_state,
                           const Tensor &qkv,
                           const Tensor &weight,
                           std::optional<Tensor> bias,
                           std::optional<Tensor> cu_seqlens,
                           std::optional<Tensor> initial_state_indices,
                           std::optional<Tensor> final_state_indices) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, conv_state, qkv, weight);
    if (final_conv_state.has_value()) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, final_conv_state.value());
    }
    if (bias.has_value()) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, bias.value());
    }
    if (cu_seqlens.has_value()) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, cu_seqlens.value());
    }
    if (initial_state_indices.has_value()) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, initial_state_indices.value());
    }
    if (final_state_indices.has_value()) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, final_state_indices.value());
    }
    INFINICORE_GRAPH_OP_DISPATCH(out->device().type(),
                                 out,
                                 conv_state,
                                 final_conv_state,
                                 qkv,
                                 weight,
                                 bias,
                                 cu_seqlens,
                                 initial_state_indices,
                                 final_state_indices);
}

void CausalConv1d::execute(Tensor out,
                           Tensor conv_state,
                           std::optional<Tensor> final_conv_state,
                           const Tensor &qkv,
                           const Tensor &weight,
                           std::optional<Tensor> bias,
                           std::optional<Tensor> cu_seqlens,
                           std::optional<Tensor> initial_state_indices,
                           std::optional<Tensor> final_state_indices) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(CausalConv1d,
                                      out,
                                      conv_state,
                                      final_conv_state,
                                      qkv,
                                      weight,
                                      bias,
                                      cu_seqlens,
                                      initial_state_indices,
                                      final_state_indices);
}

static void check_3d_tensor(const Tensor &x, const char *name, const char *shape_hint) {
    if (x->shape().size() != 3) {
        throw std::runtime_error(std::string("causal_conv1d expects ") + name + " with shape " + shape_hint);
    }
}

static size_t request_count_from_args(const Tensor &qkv, std::optional<Tensor> cu_seqlens) {
    if (cu_seqlens.has_value()) {
        const auto &cu_shape = cu_seqlens.value()->shape();
        if (cu_shape.size() != 1 || cu_shape[0] < 2) {
            throw std::runtime_error("causal_conv1d expects cu_seqlens with shape [num_requests + 1]");
        }
        return cu_shape[0] - 1;
    }
    return qkv->shape()[0];
}

static Shape final_state_shape(const Tensor &qkv,
                               const Tensor &conv_state,
                               const Tensor &weight,
                               std::optional<Tensor> cu_seqlens) {
    const auto request_count = request_count_from_args(qkv, cu_seqlens);
    return {request_count, conv_state->shape()[1], weight->shape()[2] - 1};
}

Tensor causal_conv1d(const Tensor &qkv,
                     Tensor conv_state,
                     const Tensor &weight,
                     std::optional<Tensor> bias,
                     std::optional<Tensor> cu_seqlens,
                     std::optional<Tensor> initial_state_indices,
                     std::optional<Tensor> final_state_indices) {
    check_3d_tensor(qkv, "qkv", "[B, T, C] or [1, total_tokens, C]");
    check_3d_tensor(conv_state, "conv_state", "[B/num_requests, C, state_len] or [pool_size, C, state_len]");
    check_3d_tensor(weight, "weight", "[C, 1, state_len + 1]");
    auto out = Tensor::empty(qkv->shape(), qkv->dtype(), qkv->device());
    std::optional<Tensor> final_conv_state = std::nullopt;
    if (!final_state_indices.has_value()) {
        final_conv_state = Tensor::empty(final_state_shape(qkv, conv_state, weight, cu_seqlens),
                                         conv_state->dtype(),
                                         conv_state->device());
    }
    causal_conv1d_(out,
                   conv_state,
                   final_conv_state,
                   qkv,
                   weight,
                   bias,
                   cu_seqlens,
                   initial_state_indices,
                   final_state_indices);
    return out;
}

void causal_conv1d_(Tensor out,
                    Tensor conv_state,
                    std::optional<Tensor> final_conv_state,
                    const Tensor &qkv,
                    const Tensor &weight,
                    std::optional<Tensor> bias,
                    std::optional<Tensor> cu_seqlens,
                    std::optional<Tensor> initial_state_indices,
                    std::optional<Tensor> final_state_indices) {
    check_3d_tensor(out, "out", "[B, T, C] or [1, total_tokens, C]");
    check_3d_tensor(qkv, "qkv", "[B, T, C] or [1, total_tokens, C]");
    check_3d_tensor(conv_state, "conv_state", "[B/num_requests, C, state_len] or [pool_size, C, state_len]");
    check_3d_tensor(weight, "weight", "[C, 1, state_len + 1]");
    CausalConv1d::execute(out,
                          conv_state,
                          final_conv_state,
                          qkv,
                          weight,
                          bias,
                          cu_seqlens,
                          initial_state_indices,
                          final_state_indices);
}

} // namespace infinicore::op
