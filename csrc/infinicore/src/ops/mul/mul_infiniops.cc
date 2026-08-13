#include "infinicore/ops/mul.hpp"

#ifdef ENABLE_INFINIOPS_API
#include "../infiniops_impl.hpp"

#include "base/mul.h"

namespace infinicore::op::mul_impl::infiniops {
namespace {

using TensorMeta = ::infinicore::op::infiniops::TensorMeta;

struct PlannedMeta {
    TensorMeta out, input, other;
    graph::GraphTensor out_tensor, input_tensor, other_tensor;
};

} // namespace

void *plan(Tensor out, const Tensor &input, const Tensor &other) {
    INFINICORE_ASSERT(::infinicore::op::infiniops::isSupportedDevice(out->device().type()));
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, input, other);
    return new PlannedMeta{
        TensorMeta(out),
        TensorMeta(input),
        TensorMeta(other),
        graph::GraphTensor(out),
        graph::GraphTensor(input),
        graph::GraphTensor(other)};
}

void run(void *planned_meta) {
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    infini::ops::Handle handle;
    handle.set_stream(context::getStream());
    infini::ops::Config config;
    infini::ops::Mul::Call(
        handle,
        config,
        planned->input.tensor(planned->input_tensor),
        planned->other.tensor(planned->other_tensor),
        planned->out.tensor(planned->out_tensor));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    ::infinicore::op::infiniops::registerSupportedDevices(Mul::plan_dispatcher(), &plan);
    ::infinicore::op::infiniops::registerSupportedDevices(Mul::run_dispatcher(), &run);
    ::infinicore::op::infiniops::registerSupportedDevices(Mul::cleanup_dispatcher(), &cleanup);
    return true;
}();

} // namespace infinicore::op::mul_impl::infiniops
#endif
