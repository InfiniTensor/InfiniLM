#pragma once

#include "../device.hpp"
#include "common/op.hpp"
#include <optional>

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(
    MultiheadAttention,
    Tensor,
    const Tensor &,
    const Tensor &,
    const Tensor &,
    std::optional<Tensor>,
    float,
    bool);

Tensor mha(const Tensor &q,
           const Tensor &k,
           const Tensor &v,
           std::optional<Tensor> alibi_slopes,
           float scale,
           bool is_causal);

void mha_(Tensor out,
          const Tensor &q,
          const Tensor &k,
          const Tensor &v,
          std::optional<Tensor> alibi_slopes,
          float scale,
          bool is_causal);

} // namespace infinicore::op
