#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(
    MambaSelectiveScan,
    Tensor,
    const Tensor &,
    const Tensor &,
    const Tensor &,
    const Tensor &,
    const Tensor &,
    const Tensor &,
    const Tensor &,
    const Tensor &,
    Tensor);

Tensor mamba_selective_scan(const Tensor &x,
                            const Tensor &dt,
                            const Tensor &b,
                            const Tensor &c,
                            const Tensor &a_log,
                            const Tensor &d,
                            const Tensor &gate,
                            const Tensor &dt_bias,
                            Tensor state);

void mamba_selective_scan_(Tensor out,
                           const Tensor &x,
                           const Tensor &dt,
                           const Tensor &b,
                           const Tensor &c,
                           const Tensor &a_log,
                           const Tensor &d,
                           const Tensor &gate,
                           const Tensor &dt_bias,
                           Tensor state);

} // namespace infinicore::op
