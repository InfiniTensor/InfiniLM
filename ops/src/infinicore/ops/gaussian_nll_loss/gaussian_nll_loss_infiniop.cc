#include "infinicore/ops/gaussian_nll_loss.hpp"

#include "infiniop/ops/gaussian_nll_loss.h"

#include "../infiniop_impl.hpp"

namespace infinicore::op::gaussian_nll_loss_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, GaussianNllLoss, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, out, input, target, var;
};

void *plan(Tensor out,
           const Tensor &input,
           const Tensor &target,
           const Tensor &var,
           bool full,
           double eps,
           int reduction) {
    const int full_i = full ? 1 : 0;
    size_t seed = hash_combine(out, input, target, var, full_i, eps, reduction);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, GaussianNllLoss,
        seed,
        out->desc(),
        input->desc(),
        target->desc(),
        var->desc(),
        full_i,
        eps,
        reduction);

    INFINIOP_WORKSPACE_TENSOR(workspace, GaussianNllLoss, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(out),
        graph::GraphTensor(input),
        graph::GraphTensor(target),
        graph::GraphTensor(var)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopGaussianNllLoss(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->out->data(),
        planned->input->data(),
        planned->target->data(),
        planned->var->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(GaussianNllLoss, &plan, &run, &cleanup);

} // namespace infinicore::op::gaussian_nll_loss_impl::infiniop
