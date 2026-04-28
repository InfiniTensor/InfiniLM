#include "infinicore/ops/dist.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::dist_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Dist, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, y, x1, x2;
};

void *plan(Tensor y, const Tensor &x1, const Tensor &x2, double p) {
    size_t seed = hash_combine(y, x1, x2, p);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Dist,
        seed,
        y->desc(), x1->desc(), x2->desc(), p);

    INFINIOP_WORKSPACE_TENSOR(workspace, Dist, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(y),
        graph::GraphTensor(x1),
        graph::GraphTensor(x2)};
}

void run(void *planned_meta) {
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopDist(
        p->descriptor->desc,
        p->workspace ? p->workspace->data() : nullptr,
        p->workspace ? p->workspace->numel() : 0,
        p->y->data(),
        p->x1->data(),
        p->x2->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Dist, &plan, &run, &cleanup);

} // namespace infinicore::op::dist_impl::infiniop
