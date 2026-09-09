#include "infinicore/ops/zeros.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Zeros);

Zeros::Zeros(Tensor output) {
    INFINICORE_ASSERT(output);
    INFINICORE_GRAPH_OP_DISPATCH(output->device().type(), output);
}

void Zeros::execute(Tensor output) {
    context::setDevice(output->device());
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Zeros, output);
}

void zeros_(Tensor output) {
    Zeros::execute(output);
}

} // namespace infinicore::op
