#include "infinicore/ops/paged_attention.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(PagedAttention);

PagedAttention::PagedAttention(Tensor out, const Tensor &q, const Tensor &k_cache, const Tensor &v_cache,
                               const Tensor &block_tables, const Tensor &kv_lens,
                               std::optional<Tensor> alibi_slopes, float scale) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, q, k_cache, v_cache, block_tables, kv_lens);
    INFINICORE_GRAPH_OP_DISPATCH(out->device().getType(),
                                 out, q, k_cache, v_cache, block_tables, kv_lens, alibi_slopes, scale);
}

void PagedAttention::execute(Tensor out, const Tensor &q, const Tensor &k_cache, const Tensor &v_cache,
                             const Tensor &block_tables, const Tensor &kv_lens,
                             std::optional<Tensor> alibi_slopes, float scale) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        PagedAttention,
        out, q, k_cache, v_cache, block_tables, kv_lens, alibi_slopes, scale);
}

Tensor paged_attention(const Tensor &q, const Tensor &k_cache, const Tensor &v_cache,
                       const Tensor &block_tables, const Tensor &kv_lens,
                       std::optional<Tensor> alibi_slopes, float scale) {
    auto out = Tensor::empty(q->shape(), q->dtype(), q->device());
    paged_attention_(out, q, k_cache, v_cache, block_tables, kv_lens, alibi_slopes, scale);
    return out;
}

void paged_attention_(Tensor out, const Tensor &q, const Tensor &k_cache, const Tensor &v_cache,
                      const Tensor &block_tables, const Tensor &kv_lens,
                      std::optional<Tensor> alibi_slopes, float scale) {
    PagedAttention::execute(out, q, k_cache, v_cache, block_tables, kv_lens, alibi_slopes, scale);
}

} // namespace infinicore::op
