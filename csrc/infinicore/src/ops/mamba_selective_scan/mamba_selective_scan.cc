#include "infinicore/ops/mamba_selective_scan.hpp"
#include "../../utils.hpp"

namespace infinicore::op {
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(MambaSelectiveScan);

MambaSelectiveScan::MambaSelectiveScan(Tensor out, const Tensor &x, const Tensor &dt,
                                       const Tensor &b, const Tensor &c, const Tensor &a_log,
                                       const Tensor &d, const Tensor &gate, const Tensor &dt_bias,
                                       Tensor state) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, x, dt, b, c, a_log, d, gate, dt_bias, state);
    INFINICORE_GRAPH_OP_DISPATCH(out->device().type(), out, x, dt, b, c, a_log, d, gate, dt_bias, state);
}
void MambaSelectiveScan::execute(Tensor out, const Tensor &x, const Tensor &dt,
                                 const Tensor &b, const Tensor &c, const Tensor &a_log,
                                 const Tensor &d, const Tensor &gate, const Tensor &dt_bias,
                                 Tensor state) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(MambaSelectiveScan, out, x, dt, b, c, a_log, d, gate, dt_bias, state);
}
Tensor mamba_selective_scan(const Tensor &x, const Tensor &dt, const Tensor &b, const Tensor &c, const Tensor &a_log, const Tensor &d, const Tensor &gate, const Tensor &dt_bias, Tensor state) {
    auto output = Tensor::empty(x->shape(), x->dtype(), x->device());
    mamba_selective_scan_(output, x, dt, b, c, a_log, d, gate, dt_bias, state);
    return output;
}
void mamba_selective_scan_(Tensor out, const Tensor &x, const Tensor &dt, const Tensor &b, const Tensor &c, const Tensor &a_log, const Tensor &d, const Tensor &gate, const Tensor &dt_bias, Tensor state) {
    MambaSelectiveScan::execute(out, x, dt, b, c, a_log, d, gate, dt_bias, state);
}
} // namespace infinicore::op
