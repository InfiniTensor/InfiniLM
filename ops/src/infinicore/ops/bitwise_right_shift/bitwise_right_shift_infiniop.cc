#include "infinicore/ops/bitwise_right_shift.hpp"

#include "infiniop/ops/bitwise_right_shift.h"

#include "../infiniop_impl.hpp"

namespace infinicore::op::bitwise_right_shift_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, BitwiseRightShift, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, out, input, other;
};

void *plan(Tensor out, const Tensor &input, const Tensor &other) {
    size_t seed = hash_combine(out, input, other);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, BitwiseRightShift,
        seed, out->desc(), input->desc(), other->desc());

    INFINIOP_WORKSPACE_TENSOR(workspace, BitwiseRightShift, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(out),
        graph::GraphTensor(input),
        graph::GraphTensor(other)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopBitwiseRightShift(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->out->data(),
        planned->input->data(),
        planned->other->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(BitwiseRightShift, &plan, &run, &cleanup);

} // namespace infinicore::op::bitwise_right_shift_impl::infiniop
