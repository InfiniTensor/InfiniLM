#include "infinicore/ops/gaussian_nll_loss.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(GaussianNllLoss);

GaussianNllLoss::GaussianNllLoss(Tensor out,
                                 const Tensor &input,
                                 const Tensor &target,
                                 const Tensor &var,
                                 bool full,
                                 double eps,
                                 int reduction) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, input, target, var);
    INFINICORE_GRAPH_OP_DISPATCH(out->device().getType(), out, input, target, var, full, eps, reduction);
}

void GaussianNllLoss::execute(Tensor out,
                              const Tensor &input,
                              const Tensor &target,
                              const Tensor &var,
                              bool full,
                              double eps,
                              int reduction) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(GaussianNllLoss, out, input, target, var, full, eps, reduction);
}

Tensor gaussian_nll_loss(const Tensor &input,
                         const Tensor &target,
                         const Tensor &var,
                         bool full,
                         double eps,
                         int reduction) {
    std::vector<size_t> out_shape = (reduction == 0) ? input->shape() : std::vector<size_t>{};
    auto out = Tensor::empty(out_shape, input->dtype(), input->device());
    gaussian_nll_loss_(out, input, target, var, full, eps, reduction);
    return out;
}

void gaussian_nll_loss_(Tensor out,
                        const Tensor &input,
                        const Tensor &target,
                        const Tensor &var,
                        bool full,
                        double eps,
                        int reduction) {
    GaussianNllLoss::execute(out, input, target, var, full, eps, reduction);
}

} // namespace infinicore::op
