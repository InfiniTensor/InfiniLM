#include "infinicore/ops/zeros.hpp"

#include "infinicore/context/context.hpp"

namespace infinicore::op::zeros_impl::infinirt {
namespace {

struct PlannedMeta {
    graph::GraphTensor output;
};

void *plan(Tensor output) {
    return new PlannedMeta{graph::GraphTensor(output)};
}

void run(void *planned_meta) {
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    context::setDevice(planned->output->device());
    context::setDeviceMemoryAsync(
        planned->output->data(),
        0,
        planned->output->nbytes(),
        context::getStream());
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Zeros, &plan, &run, &cleanup);

} // namespace
} // namespace infinicore::op::zeros_impl::infinirt
