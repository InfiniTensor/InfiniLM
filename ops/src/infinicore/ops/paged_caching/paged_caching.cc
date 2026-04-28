#include "infinicore/ops/paged_caching.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(PagedCaching);

PagedCaching::PagedCaching(Tensor k_cache, Tensor v_cache, const Tensor &k, const Tensor &v, const Tensor &slot_mapping) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(k_cache, v_cache, k, v, slot_mapping);
    INFINICORE_GRAPH_OP_DISPATCH(k->device().getType(), k_cache, v_cache, k, v, slot_mapping);
}

void PagedCaching::execute(Tensor k_cache, Tensor v_cache, const Tensor &k, const Tensor &v, const Tensor &slot_mapping) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(PagedCaching, k_cache, v_cache, k, v, slot_mapping);
}

void paged_caching_(Tensor k_cache, Tensor v_cache, const Tensor &k, const Tensor &v, const Tensor &slot_mapping) {
    PagedCaching::execute(k_cache, v_cache, k, v, slot_mapping);
}

} // namespace infinicore::op
