#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"
#include <optional>

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(RecurrentGatedDeltaRule,
                          Tensor,
                          Tensor,
                          std::optional<Tensor>,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          std::optional<Tensor>,
                          std::optional<Tensor>,
                          bool);

Tensor recurrent_gated_delta_rule(const Tensor &q,
                                  const Tensor &k,
                                  const Tensor &v,
                                  const Tensor &g,
                                  const Tensor &beta,
                                  const Tensor &initial_state,
                                  bool use_qk_l2norm = false);

Tensor recurrent_gated_delta_rule_indexed(const Tensor &q,
                                          const Tensor &k,
                                          const Tensor &v,
                                          const Tensor &g,
                                          const Tensor &beta,
                                          Tensor initial_state,
                                          const Tensor &initial_state_indices,
                                          const Tensor &final_state_indices,
                                          bool use_qk_l2norm = false);

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
                                 bool use_qk_l2norm = false);

} // namespace infinicore::op
