#include "infinicore/ops/mha.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(MultiheadAttention);

MultiheadAttention::MultiheadAttention(Tensor out,
                                       const Tensor &q,
                                       const Tensor &k,
                                       const Tensor &v,
                                       std::optional<Tensor> alibi_slopes,
                                       float scale,
                                       bool is_causal) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, q, k, v);
    INFINICORE_GRAPH_OP_DISPATCH(out->device().type(),
                                 out, q, k, v, alibi_slopes, scale, is_causal);
}

void MultiheadAttention::execute(Tensor out,
                                 const Tensor &q,
                                 const Tensor &k,
                                 const Tensor &v,
                                 std::optional<Tensor> alibi_slopes,
                                 float scale,
                                 bool is_causal) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        MultiheadAttention,
        out, q, k, v, alibi_slopes, scale, is_causal);
}

Tensor mha(
    const Tensor &q,
    const Tensor &k,
    const Tensor &v,
    std::optional<Tensor> alibi_slopes,
    float scale,
    bool is_causal) {
    auto out = Tensor::empty(q->shape(), q->dtype(), q->device());
    mha_(out, q, k, v, alibi_slopes, scale, is_causal);
    return out;
}

void mha_(Tensor out,
          const Tensor &q,
          const Tensor &k,
          const Tensor &v,
          std::optional<Tensor> alibi_slopes,
          float scale,
          bool is_causal) {
    MultiheadAttention::execute(out, q, k, v, alibi_slopes, scale, is_causal);
}

} // namespace infinicore::op
