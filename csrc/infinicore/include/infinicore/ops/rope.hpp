#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../nn/rope.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

namespace infinicore::op {

// QY longctx RoPE packs via cudaMemcpy*/host sync; keep eager during graph capture
// so FA/gemm can still capture around it (same pattern as MoE ATen/_dl_C).
INFINICORE_GRAPH_OP_CLASS_CAPTURE_UNSAFE(RoPE, Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, infinicore::nn::RoPE::Algo);

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
