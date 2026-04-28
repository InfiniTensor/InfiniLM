#include "infinicore/ops/add_rms_norm.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::add_rms_norm_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, AddRMSNorm, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, out, residual, a, b, weight;
    float epsilon;
};

void *plan(Tensor y, Tensor residual_out, const Tensor &a, const Tensor &b, const Tensor &weight, float epsilon) {
    size_t seed = hash_combine(y, residual_out, a, b, weight, epsilon);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, AddRMSNorm,
        seed, y->desc(), residual_out->desc(),
        a->desc(), b->desc(), weight->desc(), epsilon);

    INFINIOP_WORKSPACE_TENSOR(workspace, AddRMSNorm, descriptor);

    auto planned = new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(y),
        graph::GraphTensor(residual_out),
        graph::GraphTensor(a),
        graph::GraphTensor(b),
        graph::GraphTensor(weight),
        epsilon};

    return planned;
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopAddRMSNorm(
        planned->descriptor->desc, planned->workspace->data(), planned->workspace->numel(),
        planned->out->data(), planned->residual->data(), planned->a->data(), planned->b->data(), planned->weight->data(), context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(AddRMSNorm, &plan, &run, &cleanup);

} // namespace infinicore::op::add_rms_norm_impl::infiniop
