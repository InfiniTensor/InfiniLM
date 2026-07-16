#include "infinicore/ops/topksoftmax.hpp"

#ifdef ENABLE_INFINIOPS_API
#include "../infiniops_impl.hpp"

#include "base/topksoftmax_infinilm.h"

namespace infinicore::op::topksoftmax_impl::infiniops {
namespace {
using TensorMeta = ::infinicore::op::infiniops::TensorMeta;
struct PlannedMeta {
    TensorMeta values, indices, x;
    graph::GraphTensor values_tensor, indices_tensor, x_tensor;
    size_t topk;
    int norm;
};
} // namespace

void *plan(Tensor values, Tensor indices, const Tensor &x, const size_t topk, const int norm) {
    INFINICORE_ASSERT(::infinicore::op::infiniops::isSupportedDevice(values->device().type()));
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(values, indices, x);
    return new PlannedMeta{TensorMeta(values), TensorMeta(indices), TensorMeta(x), graph::GraphTensor(values), graph::GraphTensor(indices), graph::GraphTensor(x), topk, norm};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    infini::ops::Handle handle;
    handle.set_stream(context::getStream());
    infini::ops::Config config;
    infini::ops::TopksoftmaxInfinilm::Call(
        handle,
        config,
        planned->x.tensor(planned->x_tensor),
        static_cast<int64_t>(planned->topk),
        planned->norm != 0,
        planned->values.tensor(planned->values_tensor),
        planned->indices.tensor(planned->indices_tensor));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    ::infinicore::op::infiniops::registerSupportedDevices(Topksoftmax::plan_dispatcher(), &plan);
    ::infinicore::op::infiniops::registerSupportedDevices(Topksoftmax::run_dispatcher(), &run);
    ::infinicore::op::infiniops::registerSupportedDevices(Topksoftmax::cleanup_dispatcher(), &cleanup);
    return true;
}();
} // namespace infinicore::op::topksoftmax_impl::infiniops
#endif
