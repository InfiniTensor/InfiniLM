#include "infinicore/ops/ones.hpp"

#ifdef ENABLE_INFINIOPS_API
#include "../infiniops_impl.hpp"

#include "base/fill.h"

namespace infinicore::op::ones_impl::infiniops {
namespace {

using TensorMeta = ::infinicore::op::infiniops::TensorMeta;

struct PlannedMeta {
    TensorMeta output;
    graph::GraphTensor output_tensor;
};

void *plan(Tensor output) {
    INFINICORE_ASSERT(
        ::infinicore::op::infiniops::isSupportedDevice(output->device().type()));
    return new PlannedMeta{TensorMeta(output), graph::GraphTensor(output)};
}

void run(void *planned_meta) {
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    context::setDevice(planned->output_tensor->device());

    infini::ops::Handle handle;
    handle.set_stream(context::getStream());
    auto config = ::infinicore::op::infiniops::defaultConfigForDevice<infini::ops::Fill>(
        planned->output.device.type());
    const auto output = planned->output.tensor(planned->output_tensor);
    infini::ops::Fill::Call(handle, config, output, 1.0, output);
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    ::infinicore::op::infiniops::registerSupportedDevices(
        Ones::plan_dispatcher(), &plan);
    ::infinicore::op::infiniops::registerSupportedDevices(
        Ones::run_dispatcher(), &run);
    ::infinicore::op::infiniops::registerSupportedDevices(
        Ones::cleanup_dispatcher(), &cleanup);
    return true;
}();

} // namespace
} // namespace infinicore::op::ones_impl::infiniops
#endif
