#include "infinicore/ops/pad.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::pad_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Pad, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, y, x;
};

void *plan(Tensor y,
           const Tensor &x,
           const std::vector<int> &pad,
           const std::string &mode,
           double value) {
    size_t seed = hash_combine(y, x, mode, value, static_cast<int>(pad.size()));
    for (int v : pad) {
        hash_combine(seed, v);
    }

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Pad,
        seed,
        y->desc(),
        x->desc(),
        const_cast<int *>(pad.data()),
        pad.size() * sizeof(int),
        mode.c_str(),
        value);

    INFINIOP_WORKSPACE_TENSOR(workspace, Pad, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(y),
        graph::GraphTensor(x)};
}

void run(void *planned_meta) {
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopPad(
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

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Pad, &plan, &run, &cleanup);

} // namespace infinicore::op::pad_impl::infiniop
