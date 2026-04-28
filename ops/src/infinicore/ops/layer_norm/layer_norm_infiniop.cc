#include "infinicore/ops/layer_norm.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::layer_norm_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, LayerNorm, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, y, standardization, std_deviation, x, weight, bias;
};

void *plan(Tensor y, Tensor standardization, Tensor std_deviation, const Tensor &x, const Tensor &weight, const Tensor &bias, float epsilon) {
    size_t seed = hash_combine(y, standardization, std_deviation, x, weight, bias, epsilon);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, LayerNorm,
        seed,
        y->desc(),
        standardization->desc(),
        std_deviation->desc(),
        x->desc(),
        weight->desc(),
        bias->desc(),
        epsilon);

    INFINIOP_WORKSPACE_TENSOR(workspace, LayerNorm, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(y),
        graph::GraphTensor(standardization),
        graph::GraphTensor(std_deviation),
        graph::GraphTensor(x),
        graph::GraphTensor(weight),
        graph::GraphTensor(bias)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(
        infiniopLayerNorm(
            planned->descriptor->desc,
            planned->workspace->data(),
            planned->workspace->numel(),
            planned->y->data(),
            planned->standardization->data(),
            planned->std_deviation->data(),
            planned->x->data(),
            planned->weight->data(),
            planned->bias->data(),
            context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(LayerNorm, &plan, &run, &cleanup);

} // namespace infinicore::op::layer_norm_impl::infiniop
