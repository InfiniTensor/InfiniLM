#include "infinicore/ops/mha_kvcache.hpp"
#include "../../utils.hpp"

#ifdef ENABLE_FLASH_ATTN_DLSYM
// Lazy-registers the Hygon dlsym wrapper's plan/run/cleanup. Defined in
// libflash_attn_hygon_dlsym.so; called once per process from the dispatch
// path here (cannot be in static init due to circular dep with this lib).
extern "C" void infinicore_hygon_register_flash_attn();
#endif

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(MhaKVCache);

MhaKVCache::MhaKVCache(Tensor out,
                       const Tensor &q,
                       const Tensor &k_cache,
                       const Tensor &v_cache,
                       const Tensor &seqlens_k,
                       const Tensor &block_table,
                       std::optional<Tensor> alibi_slopes,
                       float scale) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, q, k_cache, v_cache, seqlens_k, block_table);
#ifdef ENABLE_FLASH_ATTN_DLSYM
    infinicore_hygon_register_flash_attn();
#endif
    INFINICORE_GRAPH_OP_DISPATCH(out->device().getType(),
                                 out, q, k_cache, v_cache, seqlens_k, block_table, alibi_slopes, scale);
}

void MhaKVCache::execute(Tensor out,
                         const Tensor &q,
                         const Tensor &k_cache,
                         const Tensor &v_cache,
                         const Tensor &seqlens_k,
                         const Tensor &block_table,
                         std::optional<Tensor> alibi_slopes,
                         float scale) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        MhaKVCache,
        out, q, k_cache, v_cache, seqlens_k, block_table, alibi_slopes, scale);
}

void mha_kvcache_(Tensor out,
                  const Tensor &q,
                  const Tensor &k_cache,
                  const Tensor &v_cache,
                  const Tensor &seqlens_k,
                  const Tensor &block_table,
                  std::optional<Tensor> alibi_slopes,
                  float scale) {
    MhaKVCache::execute(out, q, k_cache, v_cache, seqlens_k, block_table, alibi_slopes, scale);
}

Tensor mha_kvcache(const Tensor &q,
                   const Tensor &k_cache,
                   const Tensor &v_cache,
                   const Tensor &seqlens_k,
                   const Tensor &block_table,
                   std::optional<Tensor> alibi_slopes,
                   float scale) {
    // Output shape matches q: [batch_size, seqlen_q, num_heads, head_size]
    auto out = Tensor::empty(q->shape(), q->dtype(), q->device());
    mha_kvcache_(out, q, k_cache, v_cache, seqlens_k, block_table, alibi_slopes, scale);
    return out;
}

} // namespace infinicore::op
