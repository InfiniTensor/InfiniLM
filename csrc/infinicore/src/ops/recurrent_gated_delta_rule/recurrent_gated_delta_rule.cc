#include "infinicore/ops/recurrent_gated_delta_rule.hpp"
#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(RecurrentGatedDeltaRule);

RecurrentGatedDeltaRule::RecurrentGatedDeltaRule(Tensor out,
                                                 Tensor initial_state,
                                                 std::optional<Tensor> final_state,
                                                 const Tensor &q,
                                                 const Tensor &k,
                                                 const Tensor &v,
                                                 const Tensor &g,
                                                 const Tensor &beta,
                                                 std::optional<Tensor> initial_state_indices,
                                                 std::optional<Tensor> final_state_indices,
                                                 bool use_qk_l2norm) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, initial_state, q, k, v, g, beta);
    if (final_state.has_value()) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, final_state.value());
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
                                 initial_state_indices,
                                 final_state_indices,
                                 use_qk_l2norm);
}

void RecurrentGatedDeltaRule::execute(Tensor out,
                                      Tensor initial_state,
                                      std::optional<Tensor> final_state,
                                      const Tensor &q,
                                      const Tensor &k,
                                      const Tensor &v,
                                      const Tensor &g,
                                      const Tensor &beta,
                                      std::optional<Tensor> initial_state_indices,
                                      std::optional<Tensor> final_state_indices,
                                      bool use_qk_l2norm) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(RecurrentGatedDeltaRule,
                                      out,
                                      initial_state,
                                      final_state,
                                      q,
                                      k,
                                      v,
                                      g,
                                      beta,
                                      initial_state_indices,
                                      final_state_indices,
                                      use_qk_l2norm);
}

static Tensor ensure_4d_sequence_tensor(const Tensor &x, const char *name) {
    if (x->shape().size() == 4) {
        return x;
    }
    if (x->shape().size() == 3) {
        return x->unsqueeze(1);
    }
    throw std::runtime_error(std::string("recurrent_gated_delta_rule expects ") + name + " with shape [B, T, H, D] or [B, H, D]");
}

static Shape recurrent_output_shape(const Tensor &v) {
    const auto &shape = v->shape();
    return {shape[0], shape[1], shape[2], shape[3]};
}

Tensor recurrent_gated_delta_rule(const Tensor &q,
                                  const Tensor &k,
                                  const Tensor &v,
                                  const Tensor &g,
                                  const Tensor &beta,
                                  const Tensor &initial_state,
                                  bool use_qk_l2norm) {
    Tensor q4 = ensure_4d_sequence_tensor(q, "q");
    Tensor k4 = ensure_4d_sequence_tensor(k, "k");
    Tensor v4 = ensure_4d_sequence_tensor(v, "v");
    auto out = Tensor::empty(recurrent_output_shape(v4), v4->dtype(), v4->device());
    Shape final_state_shape = {v4->shape()[0], v4->shape()[2], v4->shape()[3], q4->shape()[3]};
    auto final_state = Tensor::empty(final_state_shape, initial_state->dtype(), initial_state->device());
    recurrent_gated_delta_rule_(out,
                                initial_state,
                                final_state,
                                q4,
                                k4,
                                v4,
                                g,
                                beta,
                                std::nullopt,
                                std::nullopt,
                                use_qk_l2norm);
    return out;
}

Tensor recurrent_gated_delta_rule_indexed(const Tensor &q,
                                          const Tensor &k,
                                          const Tensor &v,
                                          const Tensor &g,
                                          const Tensor &beta,
                                          Tensor initial_state,
                                          const Tensor &initial_state_indices,
                                          const Tensor &final_state_indices,
                                          bool use_qk_l2norm) {
    Tensor q4 = ensure_4d_sequence_tensor(q, "q");
    Tensor k4 = ensure_4d_sequence_tensor(k, "k");
    Tensor v4 = ensure_4d_sequence_tensor(v, "v");
    auto out = Tensor::empty(recurrent_output_shape(v4), v4->dtype(), v4->device());
    recurrent_gated_delta_rule_(out,
                                initial_state,
                                std::nullopt,
                                q4,
                                k4,
                                v4,
                                g,
                                beta,
                                initial_state_indices,
                                final_state_indices,
                                use_qk_l2norm);
    return out;
}

void recurrent_gated_delta_rule_(Tensor out,
                                 Tensor initial_state,
                                 std::optional<Tensor> final_state,
                                 const Tensor &q,
                                 const Tensor &k,
                                 const Tensor &v,
                                 const Tensor &g,
                                 const Tensor &beta,
                                 std::optional<Tensor> initial_state_indices,
                                 std::optional<Tensor> final_state_indices,
                                 bool use_qk_l2norm) {
    Tensor q4 = ensure_4d_sequence_tensor(q, "q");
    Tensor k4 = ensure_4d_sequence_tensor(k, "k");
    Tensor v4 = ensure_4d_sequence_tensor(v, "v");
    RecurrentGatedDeltaRule::execute(out,
                                     initial_state,
                                     final_state,
                                     q4,
                                     k4,
                                     v4,
                                     g,
                                     beta,
                                     initial_state_indices,
                                     final_state_indices,
                                     use_qk_l2norm);
}

} // namespace infinicore::op
