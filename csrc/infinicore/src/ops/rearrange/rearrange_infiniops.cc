#include "infinicore/ops/rearrange.hpp"

#ifdef ENABLE_INFINIOPS_API
#include "../infiniops_impl.hpp"

#include "base/rearrange_infinilm.h"

namespace infinicore::op::rearrange_impl::infiniops {
namespace {
using TensorMeta = ::infinicore::op::infiniops::TensorMeta;
struct PlannedMeta {
    TensorMeta y, x;
    graph::GraphTensor y_tensor, x_tensor;
};
} // namespace

void *plan(Tensor y, const Tensor &x) {
    INFINICORE_ASSERT(::infinicore::op::infiniops::isSupportedDevice(y->device().type()));
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, x);
    return new PlannedMeta{TensorMeta(y), TensorMeta(x), graph::GraphTensor(y), graph::GraphTensor(x)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    infini::ops::Handle handle;
    handle.set_stream(context::getStream());
    infini::ops::Config config;
    infini::ops::RearrangeInfinilm::Call(
        handle, config, planned->x.tensor(planned->x_tensor), planned->y.tensor(planned->y_tensor));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    ::infinicore::op::infiniops::registerSupportedDevices(Rearrange::plan_dispatcher(), &plan);
    ::infinicore::op::infiniops::registerSupportedDevices(Rearrange::run_dispatcher(), &run);
    ::infinicore::op::infiniops::registerSupportedDevices(Rearrange::cleanup_dispatcher(), &cleanup);
    return true;
}();

} // namespace infinicore::op::rearrange_impl::infiniops
#endif
