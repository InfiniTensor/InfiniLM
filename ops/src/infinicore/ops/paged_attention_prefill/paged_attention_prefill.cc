#include "infinicore/ops/paged_attention_prefill.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<PagedAttentionPrefill::schema> &PagedAttentionPrefill::dispatcher() {
    static common::OpDispatcher<PagedAttentionPrefill::schema> dispatcher_;
    return dispatcher_;
};

void PagedAttentionPrefill::execute(Tensor out, Tensor q, Tensor k_cache, Tensor v_cache,
                                    Tensor block_tables, Tensor kv_lens, Tensor cum_seqlens_q,
                                    std::optional<Tensor> alibi_slopes, float scale) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, q, k_cache, v_cache, block_tables, kv_lens, cum_seqlens_q);

    infinicore::context::setDevice(out->device());

    dispatcher().lookup(out->device().getType())(out, q, k_cache, v_cache, block_tables,
                                                 kv_lens, cum_seqlens_q, alibi_slopes, scale);
}

Tensor paged_attention_prefill(Tensor q, Tensor k_cache, Tensor v_cache,
                               Tensor block_tables, Tensor kv_lens, Tensor cum_seqlens_q,
                               std::optional<Tensor> alibi_slopes, float scale) {

    auto out = Tensor::empty(q->shape(), q->dtype(), q->device());
    paged_attention_prefill_(out, q, k_cache, v_cache, block_tables, kv_lens, cum_seqlens_q, alibi_slopes, scale);
    return out;
}

void paged_attention_prefill_(Tensor out, Tensor q, Tensor k_cache, Tensor v_cache,
                              Tensor block_tables, Tensor kv_lens, Tensor cum_seqlens_q,
                              std::optional<Tensor> alibi_slopes, float scale) {

    PagedAttentionPrefill::execute(out, q, k_cache, v_cache, block_tables, kv_lens, cum_seqlens_q, alibi_slopes, scale);
}

} // namespace infinicore::op
