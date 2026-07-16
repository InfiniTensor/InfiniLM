#include "infinicore/ops/mha_varlen.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(MultiheadAttentionVarlen);

MultiheadAttentionVarlen::MultiheadAttentionVarlen(Tensor out,
                                                   const Tensor &q,
                                                   const Tensor &k,
                                                   const Tensor &v,
                                                   const Tensor &cum_seqlens_q,
                                                   const Tensor &cum_seqlens_kv,
                                                   std::optional<Tensor> block_table,
                                                   int max_seqlen_q,
                                                   int max_seqlen_k,
                                                   std::optional<Tensor> alibi_slopes,
                                                   float scale) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, q, k, v, cum_seqlens_q, cum_seqlens_kv);
    if (block_table.has_value()) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, block_table.value());
    }
    INFINICORE_GRAPH_OP_DISPATCH(out->device().type(),
                                 out, q, k, v, cum_seqlens_q, cum_seqlens_kv, block_table, max_seqlen_q, max_seqlen_k, alibi_slopes, scale);
}

void MultiheadAttentionVarlen::execute(Tensor out,
                                       const Tensor &q,
                                       const Tensor &k,
                                       const Tensor &v,
                                       const Tensor &cum_seqlens_q,
                                       const Tensor &cum_seqlens_kv,
                                       std::optional<Tensor> block_table,
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
    std::optional<Tensor> block_table,
    int max_seqlen_q,
    int max_seqlen_k,
    std::optional<Tensor> alibi_slopes,
    float scale) {
    auto out_shape = q->shape();
    out_shape.back() = v->shape().back();
    auto out = Tensor::empty(out_shape, q->dtype(), q->device());
    mha_varlen_(out, q, k, v, cum_seqlens_q, cum_seqlens_kv, block_table, max_seqlen_q, max_seqlen_k, alibi_slopes, scale);
    return out;
}

void mha_varlen_(Tensor out,
                 const Tensor &q,
                 const Tensor &k,
                 const Tensor &v,
                 const Tensor &cum_seqlens_q,
                 const Tensor &cum_seqlens_kv,
                 std::optional<Tensor> block_table,
                 int max_seqlen_q,
                 int max_seqlen_k,
                 std::optional<Tensor> alibi_slopes,
                 float scale) {
    MultiheadAttentionVarlen::execute(out, q, k, v, cum_seqlens_q, cum_seqlens_kv, block_table, max_seqlen_q, max_seqlen_k, alibi_slopes, scale);
}

} // namespace infinicore::op
