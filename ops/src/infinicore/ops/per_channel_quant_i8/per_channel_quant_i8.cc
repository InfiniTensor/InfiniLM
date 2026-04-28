#include "infinicore/ops/per_channel_quant_i8.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(PerChannelQuantI8);

PerChannelQuantI8::PerChannelQuantI8(const Tensor &x, Tensor x_packed, Tensor x_scale) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(x, x_packed, x_scale);
    INFINICORE_GRAPH_OP_DISPATCH(x->device().getType(), x, x_packed, x_scale);
}

void PerChannelQuantI8::execute(const Tensor &x, Tensor x_packed, Tensor x_scale) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(PerChannelQuantI8, x, x_packed, x_scale);
}

void per_channel_quant_i8_(const Tensor &x, Tensor x_packed, Tensor x_scale) {
    PerChannelQuantI8::execute(x, x_packed, x_scale);
}
} // namespace infinicore::op
