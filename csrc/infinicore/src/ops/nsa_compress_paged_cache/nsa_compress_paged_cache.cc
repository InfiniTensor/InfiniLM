#include "infinicore/ops/nsa_compress_paged_cache.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(NsaCompressPagedCache);

NsaCompressPagedCache::NsaCompressPagedCache(Tensor k_cmp, Tensor v_cmp, const Tensor &k_cache, const Tensor &v_cache,
                                             const Tensor &block_tables, const Tensor &kv_lens, int nsa_block_size,
                                             bool update_last_only) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(k_cmp, v_cmp, k_cache, v_cache, block_tables, kv_lens);
    INFINICORE_GRAPH_OP_DISPATCH(k_cmp->device().type(), k_cmp, v_cmp, k_cache, v_cache, block_tables, kv_lens, nsa_block_size, update_last_only);
}

void NsaCompressPagedCache::execute(Tensor k_cmp, Tensor v_cmp, const Tensor &k_cache, const Tensor &v_cache,
                                    const Tensor &block_tables, const Tensor &kv_lens, int nsa_block_size,
                                    bool update_last_only) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        NsaCompressPagedCache,
        k_cmp, v_cmp, k_cache, v_cache, block_tables, kv_lens, nsa_block_size, update_last_only);
}

void nsa_compress_paged_cache_(Tensor k_cmp, Tensor v_cmp, const Tensor &k_cache, const Tensor &v_cache,
                               const Tensor &block_tables, const Tensor &kv_lens, int nsa_block_size,
                               bool update_last_only) {
    NsaCompressPagedCache::execute(k_cmp, v_cmp, k_cache, v_cache, block_tables, kv_lens, nsa_block_size, update_last_only);
}

} // namespace infinicore::op
