#include "infinicore/ops/rope.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(RoPE);

RoPE::RoPE(Tensor x_out,
           const Tensor &x,
           const Tensor &pos,
           const Tensor &sin_table,
           const Tensor &cos_table,
           infinicore::nn::RoPE::Algo algo) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(x_out, x, pos, sin_table, cos_table);
    INFINICORE_GRAPH_OP_DISPATCH(x_out->device().getType(), x_out, x, pos, sin_table, cos_table, algo);
}

void RoPE::execute(Tensor x_out,
                   const Tensor &x,
                   const Tensor &pos,
                   const Tensor &sin_table,
                   const Tensor &cos_table,
                   infinicore::nn::RoPE::Algo algo) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(RoPE, x_out, x, pos, sin_table, cos_table, algo);
}

void rope_(Tensor x_out,
           const Tensor &x,
           const Tensor &pos,
           const Tensor &sin_table,
           const Tensor &cos_table,
           infinicore::nn::RoPE::Algo algo) {
    RoPE::execute(x_out, x, pos, sin_table, cos_table, algo);
}

Tensor rope(const Tensor &x,
            const Tensor &pos,
            const Tensor &sin_table,
            const Tensor &cos_table,
            infinicore::nn::RoPE::Algo algo) {
    auto x_out = Tensor::empty(x->shape(), x->dtype(), x->device());
    rope_(x_out, x, pos, sin_table, cos_table, algo);
    return x_out;
}

} // namespace infinicore::op
