#include "infinicore/ops/prelu.hpp"

#include "infiniop/ops/prelu.h"

#include "../infiniop_impl.hpp"

namespace infinicore::op::prelu_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Prelu, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, out, input, weight;
};

void *plan(Tensor out, const Tensor &input, const Tensor &weight) {
    size_t seed = hash_combine(out, input, weight);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Prelu,
        seed,
        out->desc(),
        input->desc(),
        weight->desc());

    INFINIOP_WORKSPACE_TENSOR(workspace, Prelu, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(out),
        graph::GraphTensor(input),
        graph::GraphTensor(weight)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopPrelu(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->out->data(),
        planned->input->data(),
        planned->weight->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Prelu, &plan, &run, &cleanup);

} // namespace infinicore::op::prelu_impl::infiniop
