#include "infinicore/ops/chunk_gated_delta_rule.hpp"
#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(ChunkGatedDeltaRule);

ChunkGatedDeltaRule::ChunkGatedDeltaRule(Tensor out,
                                         Tensor initial_state,
                                         std::optional<Tensor> final_state,
                                         const Tensor &q,
                                         const Tensor &k,
                                         const Tensor &v,
                                         const Tensor &g,
                                         const Tensor &beta,
                                         std::optional<Tensor> cu_seqlens,
                                         std::optional<Tensor> initial_state_indices,
                                         std::optional<Tensor> final_state_indices,
                                         bool use_qk_l2norm,
                                         size_t chunk_size) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, initial_state, q, k, v, g, beta);
    if (final_state.has_value()) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, final_state.value());
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
                                 initial_state,
                                 final_state,
                                 q,
                                 k,
                                 v,
                                 g,
                                 beta,
                                 cu_seqlens,
                                 initial_state_indices,
                                 final_state_indices,
                                 use_qk_l2norm,
                                 chunk_size);
}

void ChunkGatedDeltaRule::execute(Tensor out,
                                  Tensor initial_state,
                                  std::optional<Tensor> final_state,
                                  const Tensor &q,
                                  const Tensor &k,
                                  const Tensor &v,
                                  const Tensor &g,
                                  const Tensor &beta,
                                  std::optional<Tensor> cu_seqlens,
                                  std::optional<Tensor> initial_state_indices,
                                  std::optional<Tensor> final_state_indices,
                                  bool use_qk_l2norm,
                                  size_t chunk_size) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(ChunkGatedDeltaRule,
                                      out,
                                      initial_state,
                                      final_state,
                                      q,
                                      k,
                                      v,
                                      g,
                                      beta,
                                      cu_seqlens,
                                      initial_state_indices,
                                      final_state_indices,
                                      use_qk_l2norm,
                                      chunk_size);
}

static void check_4d_sequence_tensor(const Tensor &x, const char *name) {
    if (x->shape().size() != 4) {
        throw std::runtime_error(std::string("chunk_gated_delta_rule expects ") + name + " with shape [B, T, H, D] or [1, total_tokens, H, D]");
    }
}

static Shape chunk_final_state_shape(const Tensor &q,
                                     const Tensor &v,
                                     std::optional<Tensor> cu_seqlens) {
    const auto &q_shape = q->shape();
    const auto &v_shape = v->shape();
    size_t B = cu_seqlens.has_value() ? cu_seqlens.value()->shape()[0] - 1 : v_shape[0];
    size_t Hv = v_shape[2];
    size_t Dk = q_shape[3];
    size_t Dv = v_shape[3];
    return {B, Hv, Dv, Dk};
}

Tensor chunk_gated_delta_rule(const Tensor &q,
                              const Tensor &k,
                              const Tensor &v,
                              const Tensor &g,
                              const Tensor &beta,
                              Tensor initial_state,
                              std::optional<Tensor> cu_seqlens,
                              std::optional<Tensor> initial_state_indices,
                              std::optional<Tensor> final_state_indices,
                              bool use_qk_l2norm,
                              size_t chunk_size) {
    check_4d_sequence_tensor(q, "q");
    check_4d_sequence_tensor(k, "k");
    check_4d_sequence_tensor(v, "v");
    auto out = Tensor::empty(v->shape(), v->dtype(), v->device());
    std::optional<Tensor> final_state = std::nullopt;
    if (!final_state_indices.has_value()) {
        final_state = Tensor::empty(chunk_final_state_shape(q, v, cu_seqlens),
                                    initial_state->dtype(),
                                    initial_state->device());
    }
    chunk_gated_delta_rule_(out,
                            initial_state,
                            final_state,
                            q,
                            k,
                            v,
                            g,
                            beta,
                            cu_seqlens,
                            initial_state_indices,
                            final_state_indices,
                            use_qk_l2norm,
                            chunk_size);
    return out;
}

void chunk_gated_delta_rule_(Tensor out,
                             Tensor initial_state,
                             std::optional<Tensor> final_state,
                             const Tensor &q,
                             const Tensor &k,
                             const Tensor &v,
                             const Tensor &g,
                             const Tensor &beta,
                             std::optional<Tensor> cu_seqlens,
                             std::optional<Tensor> initial_state_indices,
                             std::optional<Tensor> final_state_indices,
                             bool use_qk_l2norm,
                             size_t chunk_size) {
    check_4d_sequence_tensor(q, "q");
    check_4d_sequence_tensor(k, "k");
    check_4d_sequence_tensor(v, "v");
    ChunkGatedDeltaRule::execute(out,
                                 initial_state,
                                 final_state,
                                 q,
                                 k,
                                 v,
                                 g,
                                 beta,
                                 cu_seqlens,
                                 initial_state_indices,
                                 final_state_indices,
                                 use_qk_l2norm,
                                 chunk_size);
}

} // namespace infinicore::op
