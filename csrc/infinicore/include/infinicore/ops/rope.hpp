#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../nn/rope.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(RoPE, Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, infinicore::nn::RoPE::Algo);

// Internal
void rope_(Tensor x_out,
           const Tensor &x,
           const Tensor &pos,
           const Tensor &sin_table,
           const Tensor &cos_table,
           infinicore::nn::RoPE::Algo algo);

// Public API
Tensor rope(const Tensor &x,
            const Tensor &pos,
            const Tensor &sin_table,
            const Tensor &cos_table,
            infinicore::nn::RoPE::Algo algo);

} // namespace infinicore::op
