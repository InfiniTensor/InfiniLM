#include "infinicore/ops/diff.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::diff_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Diff, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, y, x;
};

void *plan(Tensor y, const Tensor &x, int dim, int n) {
    size_t seed = hash_combine(y, x, dim, n);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Diff,
        seed,
        y->desc(), x->desc(), dim, n);

    INFINIOP_WORKSPACE_TENSOR(workspace, Diff, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(y),
        graph::GraphTensor(x)};
}

void run(void *planned_meta) {
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopDiff(
        p->descriptor->desc,
        p->workspace ? p->workspace->data() : nullptr,
        p->workspace ? p->workspace->numel() : 0,
        p->y->data(),
        p->x->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Diff, &plan, &run, &cleanup);

} // namespace infinicore::op::diff_impl::infiniop
