#include "infinicore/ops/add_rms_norm.hpp"

#ifdef ENABLE_INFINIOPS_API
#include "../infiniops_impl.hpp"

#include "base/add_rms_norm.h"

#include <optional>

namespace infinicore::op::add_rms_norm_impl::infiniops {
namespace {

using TensorMeta = ::infinicore::op::infiniops::TensorMeta;

struct PlannedMeta {
    TensorMeta out, residual, a, b, weight;
    graph::GraphTensor out_tensor, residual_tensor, a_tensor, b_tensor, weight_tensor;
    float epsilon;
};

} // namespace

void *plan(Tensor out, Tensor residual, const Tensor &a, const Tensor &b, const Tensor &weight, float epsilon) {
    INFINICORE_ASSERT(::infinicore::op::infiniops::isSupportedDevice(out->device().type()));
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, residual, a, b, weight);

    return new PlannedMeta{
        TensorMeta(out),
        TensorMeta(residual),
        TensorMeta(a),
        TensorMeta(b),
        TensorMeta(weight),
        graph::GraphTensor(out),
        graph::GraphTensor(residual),
        graph::GraphTensor(a),
        graph::GraphTensor(b),
        graph::GraphTensor(weight),
        epsilon};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    infini::ops::Handle handle;
    handle.set_stream(context::getStream());
    infini::ops::Config config;

    infini::ops::AddRmsNorm::Call(
        handle,
        config,
        planned->a.tensor(planned->a_tensor),
        planned->b.tensor(planned->b_tensor),
        planned->weight.tensor(planned->weight_tensor),
        std::optional<float>{planned->epsilon},
        planned->out.tensor(planned->out_tensor),
        planned->residual.tensor(planned->residual_tensor));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    ::infinicore::op::infiniops::registerSupportedDevices(AddRMSNorm::plan_dispatcher(), &plan);
    ::infinicore::op::infiniops::registerSupportedDevices(AddRMSNorm::run_dispatcher(), &run);
    ::infinicore::op::infiniops::registerSupportedDevices(AddRMSNorm::cleanup_dispatcher(), &cleanup);
    return true;
}();

} // namespace infinicore::op::add_rms_norm_impl::infiniops
#endif
