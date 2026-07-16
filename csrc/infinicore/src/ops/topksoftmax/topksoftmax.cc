#include "infinicore/ops/topksoftmax.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Topksoftmax);

Topksoftmax::Topksoftmax(Tensor values,
                         Tensor indices,
                         const Tensor &x,
                         const size_t topk,
                         const int norm) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(values, indices, x);
    INFINICORE_GRAPH_OP_DISPATCH(values->device().type(), values, indices, x, topk, norm);
}

void Topksoftmax::execute(Tensor values,
                          Tensor indices,
                          const Tensor &x,
                          const size_t topk,
                          const int norm) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Topksoftmax, values, indices, x, topk, norm);
}

void topksoftmax(Tensor values, Tensor indices, const Tensor &x, const size_t topk, const int norm) {
    Topksoftmax::execute(values, indices, x, topk, norm);
}

} // namespace infinicore::op
