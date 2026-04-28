#include "infinicore/ops/silu_and_mul.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(SiluAndMul);

SiluAndMul::SiluAndMul(Tensor out, const Tensor &x) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, x);
    INFINICORE_GRAPH_OP_DISPATCH(out->device().getType(), out, x);
}

void SiluAndMul::execute(Tensor out, const Tensor &x) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(SiluAndMul, out, x);
}

Tensor silu_and_mul(const Tensor &x) {
    Shape shape = x->shape();
    size_t ndim = x->ndim();

    if (shape[ndim - 1] % 2 != 0) {
        throw std::runtime_error("SiluAndMul input last dim must be even.");
    }
    shape[ndim - 1] /= 2;

    auto out = Tensor::empty(shape, x->dtype(), x->device());
    silu_and_mul_(out, x);
    return out;
}

void silu_and_mul_(Tensor out, const Tensor &x) {
    SiluAndMul::execute(out, x);
}

} // namespace infinicore::op
