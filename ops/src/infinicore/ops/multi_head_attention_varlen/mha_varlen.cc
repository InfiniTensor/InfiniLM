#include "infinicore/ops/mha_varlen.hpp"
#include "../../utils.hpp"

#ifdef ENABLE_FLASH_ATTN_DLSYM
extern "C" void infinicore_hygon_register_flash_attn();
#endif

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(MultiheadAttentionVarlen);

MultiheadAttentionVarlen::MultiheadAttentionVarlen(Tensor out,
                                                   const Tensor &q,
                                                   const Tensor &k,
                                                   const Tensor &v,
                                                   const Tensor &cum_seqlens_q,
                                                   const Tensor &cum_seqlens_kv,
                                                   const Tensor &block_table,
                                                   int max_seqlen_q,
                                                   int max_seqlen_k,
                                                   std::optional<Tensor> alibi_slopes,
                                                   float scale) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, q, k, v, cum_seqlens_q, cum_seqlens_kv, block_table);
#ifdef ENABLE_FLASH_ATTN_DLSYM
    infinicore_hygon_register_flash_attn();
#endif
    INFINICORE_GRAPH_OP_DISPATCH(out->device().getType(),
                                 out, q, k, v, cum_seqlens_q, cum_seqlens_kv, block_table, max_seqlen_q, max_seqlen_k, alibi_slopes, scale);
}

void MultiheadAttentionVarlen::execute(Tensor out,
                                       const Tensor &q,
                                       const Tensor &k,
                                       const Tensor &v,
                                       const Tensor &cum_seqlens_q,
                                       const Tensor &cum_seqlens_kv,
                                       const Tensor &block_table,
                                       int max_seqlen_q,
                                       int max_seqlen_k,
                                       std::optional<Tensor> alibi_slopes,
                                       float scale) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        MultiheadAttentionVarlen,
        out, q, k, v, cum_seqlens_q, cum_seqlens_kv, block_table, max_seqlen_q, max_seqlen_k, alibi_slopes, scale);
}

Tensor mha_varlen(
    const Tensor &q,
    const Tensor &k,
    const Tensor &v,
    const Tensor &cum_seqlens_q,
    const Tensor &cum_seqlens_kv,
    const Tensor &block_table,
    int max_seqlen_q,
    int max_seqlen_k,
    std::optional<Tensor> alibi_slopes,
    float scale) {
    auto out = Tensor::empty(q->shape(), q->dtype(), q->device());
    mha_varlen_(out, q, k, v, cum_seqlens_q, cum_seqlens_kv, block_table, max_seqlen_q, max_seqlen_k, alibi_slopes, scale);
    return out;
}

void mha_varlen_(Tensor out,
                 const Tensor &q,
                 const Tensor &k,
                 const Tensor &v,
                 const Tensor &cum_seqlens_q,
                 const Tensor &cum_seqlens_kv,
                 const Tensor &block_table,
                 int max_seqlen_q,
                 int max_seqlen_k,
                 std::optional<Tensor> alibi_slopes,
                 float scale) {
    MultiheadAttentionVarlen::execute(out, q, k, v, cum_seqlens_q, cum_seqlens_kv, block_table, max_seqlen_q, max_seqlen_k, alibi_slopes, scale);
}

} // namespace infinicore::op
