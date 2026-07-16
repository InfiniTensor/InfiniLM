#include "infinicore/ops/nsa_paged_attention.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(NsaPagedAttention);

NsaPagedAttention::NsaPagedAttention(Tensor out, const Tensor &q, const Tensor &k_cmp, const Tensor &v_cmp,
                                     const Tensor &k_cache, const Tensor &v_cache, const Tensor &block_tables,
                                     const Tensor &kv_lens, const Tensor &gates, float scale, int nsa_block_size,
                                     int window_size, int select_blocks) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, q, k_cmp, v_cmp, k_cache, v_cache, block_tables, kv_lens, gates);
    INFINICORE_GRAPH_OP_DISPATCH(out->device().type(),
                                 out, q, k_cmp, v_cmp, k_cache, v_cache, block_tables, kv_lens, gates, scale, nsa_block_size, window_size, select_blocks);
}

void NsaPagedAttention::execute(Tensor out, const Tensor &q, const Tensor &k_cmp, const Tensor &v_cmp,
                                const Tensor &k_cache, const Tensor &v_cache, const Tensor &block_tables,
                                const Tensor &kv_lens, const Tensor &gates, float scale, int nsa_block_size,
                                int window_size, int select_blocks) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        NsaPagedAttention,
        out, q, k_cmp, v_cmp, k_cache, v_cache, block_tables, kv_lens, gates, scale, nsa_block_size, window_size, select_blocks);
}

Tensor nsa_paged_attention(const Tensor &q, const Tensor &k_cmp, const Tensor &v_cmp, const Tensor &k_cache, const Tensor &v_cache,
                           const Tensor &block_tables, const Tensor &kv_lens, const Tensor &gates,
                           float scale, int nsa_block_size, int window_size, int select_blocks) {
    auto out = Tensor::empty(q->shape(), q->dtype(), q->device());
    nsa_paged_attention_(out, q, k_cmp, v_cmp, k_cache, v_cache, block_tables, kv_lens, gates, scale, nsa_block_size, window_size, select_blocks);
    return out;
}

void nsa_paged_attention_(Tensor out, const Tensor &q, const Tensor &k_cmp, const Tensor &v_cmp, const Tensor &k_cache, const Tensor &v_cache,
                          const Tensor &block_tables, const Tensor &kv_lens, const Tensor &gates,
                          float scale, int nsa_block_size, int window_size, int select_blocks) {
    NsaPagedAttention::execute(out, q, k_cmp, v_cmp, k_cache, v_cache, block_tables, kv_lens, gates, scale, nsa_block_size, window_size, select_blocks);
}

} // namespace infinicore::op
