#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(MRoPE,
                          Tensor,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          int,
                          int,
                          int,
                          int,
                          int,
                          bool);

void mrope_(Tensor q_out,
            Tensor k_out,
            const Tensor &q,
            const Tensor &k,
            const Tensor &cos,
            const Tensor &sin,
            const Tensor &positions,
            int head_size,
            int rotary_dim,
            int section_t,
            int section_h,
            int section_w,
            bool interleaved);

std::pair<Tensor, Tensor> mrope(const Tensor &q,
                                const Tensor &k,
                                const Tensor &cos,
                                const Tensor &sin,
                                const Tensor &positions,
                                int head_size,
                                int rotary_dim,
                                int section_t,
                                int section_h,
                                int section_w,
                                bool interleaved);

} // namespace infinicore::op
