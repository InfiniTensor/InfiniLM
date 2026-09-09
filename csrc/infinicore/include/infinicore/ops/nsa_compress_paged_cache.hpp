#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(NsaCompressPagedCache, Tensor, Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int, bool);

void nsa_compress_paged_cache_(Tensor k_cmp, Tensor v_cmp, const Tensor &k_cache, const Tensor &v_cache,
                               const Tensor &block_tables, const Tensor &kv_lens, int nsa_block_size,
                               bool update_last_only = false);

} // namespace infinicore::op
