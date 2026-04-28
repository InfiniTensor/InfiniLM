#include "../infiniop_impl.hpp"
#include "infinicore/ops/gemm.hpp"

namespace infinicore::op::gemm_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Gemm, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, c, a, b;
    float alpha, beta;
};

void *plan(Tensor c, const Tensor &a, const Tensor &b, float alpha, float beta) {
    size_t seed = hash_combine(c, a, b);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Gemm,
        seed, c->desc(), a->desc(), b->desc());

    INFINIOP_WORKSPACE_TENSOR(workspace, Gemm, descriptor);

    auto planned = new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(c),
        graph::GraphTensor(a),
        graph::GraphTensor(b),
        alpha, beta};

    return planned;
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopGemm(
        planned->descriptor->desc, planned->workspace->data(), planned->workspace->numel(),
        planned->c->data(), planned->a->data(), planned->b->data(), planned->alpha, planned->beta, context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Gemm, &plan, &run, &cleanup);

} // namespace infinicore::op::gemm_impl::infiniop
