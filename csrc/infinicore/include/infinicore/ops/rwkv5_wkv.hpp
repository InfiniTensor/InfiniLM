#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(
    Rwkv5Wkv,
    Tensor,
    const Tensor &,
    const Tensor &,
    const Tensor &,
    const Tensor &,
    const Tensor &,
    Tensor);

Tensor rwkv5_wkv(const Tensor &receptance,
                 const Tensor &key,
                 const Tensor &value,
                 const Tensor &time_decay,
                 const Tensor &time_faaaa,
                 Tensor state);

void rwkv5_wkv_(Tensor out,
                const Tensor &receptance,
                const Tensor &key,
                const Tensor &value,
                const Tensor &time_decay,
                const Tensor &time_faaaa,
                Tensor state);

} // namespace infinicore::op
