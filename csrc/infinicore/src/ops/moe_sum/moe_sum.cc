#include "infinicore/ops/moe_sum.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(MoeSum);

MoeSum::MoeSum(Tensor output, const Tensor &input) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);
    INFINICORE_GRAPH_OP_DISPATCH(output->device().type(), output, input);
}

void MoeSum::execute(Tensor output, const Tensor &input) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(MoeSum, output, input);
}

Tensor moe_sum(const Tensor &input) {
    auto shape = input->shape();
    INFINICORE_ASSERT(shape.size() == 3);
    auto output = Tensor::empty({shape[0], shape[2]}, input->dtype(), input->device());
    moe_sum_(output, input);
    return output;
}

void moe_sum_(Tensor output, const Tensor &input) {
    MoeSum::execute(output, input);
}

} // namespace infinicore::op
