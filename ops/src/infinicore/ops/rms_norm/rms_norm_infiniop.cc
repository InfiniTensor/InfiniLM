#include "infinicore/ops/rms_norm.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::rms_norm_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, RMSNorm, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, y, x, weight;
};

void *plan(Tensor y, const Tensor &x, const Tensor &weight, float epsilon) {
    size_t seed = hash_combine(y, x, weight, epsilon);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, RMSNorm,
        seed, y->desc(),
        x->desc(),
        weight->desc(),
        epsilon);

    INFINIOP_WORKSPACE_TENSOR(workspace, RMSNorm, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(y),
        graph::GraphTensor(x),
        graph::GraphTensor(weight)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(
        infiniopRMSNorm(
            planned->descriptor->desc,
            planned->workspace->data(),
            planned->workspace->numel(),
            planned->y->data(),
            planned->x->data(),
            planned->weight->data(),
            context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(RMSNorm, &plan, &run, &cleanup);

} // namespace infinicore::op::rms_norm_impl::infiniop
