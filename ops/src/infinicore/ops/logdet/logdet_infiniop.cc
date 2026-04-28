#include "infinicore/ops/logdet.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::logdet_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Logdet, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, y, x;
};

void *plan(Tensor y, const Tensor &x) {
    size_t seed = hash_combine(y, x);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Logdet,
        seed,
        y->desc(), x->desc());

    INFINIOP_WORKSPACE_TENSOR(workspace, Logdet, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(y),
        graph::GraphTensor(x)};
}

void run(void *planned_meta) {
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopLogdet(
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

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Logdet, &plan, &run, &cleanup);

} // namespace infinicore::op::logdet_impl::infiniop
