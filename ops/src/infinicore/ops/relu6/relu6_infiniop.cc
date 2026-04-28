#include "infinicore/ops/relu6.hpp"

#include "infiniop/ops/relu6.h"

#include "../infiniop_impl.hpp"

namespace infinicore::op::relu6_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Relu6, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, out, input;
};

void *plan(Tensor out, const Tensor &input) {
    size_t seed = hash_combine(out, input);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Relu6,
        seed,
        out->desc(),
        input->desc());

    INFINIOP_WORKSPACE_TENSOR(workspace, Relu6, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(out),
        graph::GraphTensor(input)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopRelu6(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->out->data(),
        planned->input->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Relu6, &plan, &run, &cleanup);

} // namespace infinicore::op::relu6_impl::infiniop
