#include "infinicore/ops/kv_caching.hpp"

#include "../../utils.hpp"

namespace infinicore::op {
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(KVCaching);

KVCaching::KVCaching(Tensor k_cache,
                     Tensor v_cache,
                     const Tensor &k,
                     const Tensor &v,
                     const Tensor &past_kv_lengths) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(k_cache, v_cache, k, v, past_kv_lengths);
    INFINICORE_GRAPH_OP_DISPATCH(k_cache->device().getType(),
                                 k_cache,
                                 v_cache,
                                 k,
                                 v,
                                 past_kv_lengths);
}

void KVCaching::execute(Tensor k_cache,
                        Tensor v_cache,
                        const Tensor &k,
                        const Tensor &v,
                        const Tensor &past_kv_lengths) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(KVCaching,
                                      k_cache,
                                      v_cache,
                                      k,
                                      v,
                                      past_kv_lengths);
}

void kv_caching_(Tensor k_cache,
                 Tensor v_cache,
                 const Tensor &k,
                 const Tensor &v,
                 const Tensor &past_kv_lengths) {
    KVCaching::execute(k_cache, v_cache, k, v, past_kv_lengths);
}
} // namespace infinicore::op
