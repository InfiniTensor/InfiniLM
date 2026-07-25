#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(PagedCaching, Tensor, Tensor, const Tensor &, const Tensor &, const Tensor &);

void paged_caching_(Tensor k_cache, Tensor v_cache, const Tensor &k, const Tensor &v, const Tensor &slot_mapping);

} // namespace infinicore::op
