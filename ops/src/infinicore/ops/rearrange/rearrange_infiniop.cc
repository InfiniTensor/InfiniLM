#include "infinicore/ops/rearrange.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::rearrange_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Rearrange, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor y, x;
};

void *plan(Tensor y, const Tensor &x) {
    size_t seed = hash_combine(y, x);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Rearrange,
        seed, y->desc(),
        x->desc());

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(y),
        graph::GraphTensor(x)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(
        infiniopRearrange(
            planned->descriptor->desc,
            planned->y->data(),
            planned->x->data(),
            context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Rearrange, &plan, &run, &cleanup);

} // namespace infinicore::op::rearrange_impl::infiniop
