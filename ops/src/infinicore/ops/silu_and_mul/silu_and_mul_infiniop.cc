#include "../infiniop_impl.hpp"
#include "infinicore/ops/silu_and_mul.hpp"

namespace infinicore::op::silu_and_mul_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, SiluAndMul, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, output, input;
};

void *plan(Tensor output, const Tensor &input) {
    size_t seed = hash_combine(output, input);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, SiluAndMul,
        seed, output->desc(), input->desc());

    INFINIOP_WORKSPACE_TENSOR(workspace, SiluAndMul, descriptor);

    auto planned = new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(output),
        graph::GraphTensor(input)};

    return planned;
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopSiluAndMul(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->output->data(),
        planned->input->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(SiluAndMul, &plan, &run, &cleanup);

} // namespace infinicore::op::silu_and_mul_impl::infiniop
