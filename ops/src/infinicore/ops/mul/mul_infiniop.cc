#include "infinicore/ops/mul.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::mul_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Mul, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, c, a, b;
};

void *plan(Tensor c, const Tensor &a, const Tensor &b) {
    size_t seed = hash_combine(c, b, a);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Mul,
        seed, c->desc(), a->desc(), b->desc());

    INFINIOP_WORKSPACE_TENSOR(workspace, Mul, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(c),
        graph::GraphTensor(a),
        graph::GraphTensor(b)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopMul(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->c->data(),
        planned->a->data(),
        planned->b->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Mul, &plan, &run, &cleanup);

} // namespace infinicore::op::mul_impl::infiniop
