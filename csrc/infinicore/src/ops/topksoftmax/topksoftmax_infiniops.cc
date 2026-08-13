#include "infinicore/ops/topksoftmax.hpp"

#ifdef ENABLE_INFINIOPS_API
#include "../infiniops_impl.hpp"

#include "base/topk_softmax.h"

#include <optional>

namespace infinicore::op::topksoftmax_impl::infiniops {
namespace {
using TensorMeta = ::infinicore::op::infiniops::TensorMeta;
struct PlannedMeta {
    TensorMeta values, indices, token_expert_indices, x;
    graph::GraphTensor values_tensor, indices_tensor, token_expert_indices_tensor, x_tensor;
    int norm;
    Tensor token_expert_indices_owner;
};
} // namespace

void *plan(Tensor values, Tensor indices, const Tensor &x, const size_t topk, const int norm) {
    INFINICORE_ASSERT(::infinicore::op::infiniops::isSupportedDevice(values->device().type()));
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(values, indices, x);
    auto token_expert_indices = Tensor::empty({x->size(0), topk}, DataType::kInt32, indices->device());
    return new PlannedMeta{
        TensorMeta(values),
        TensorMeta(indices),
        TensorMeta(token_expert_indices),
        TensorMeta(x),
        graph::GraphTensor(values),
        graph::GraphTensor(indices),
        graph::GraphTensor(token_expert_indices),
        graph::GraphTensor(x),
        norm,
        token_expert_indices};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    infini::ops::Handle handle;
    handle.set_stream(context::getStream());
    infini::ops::Config config;
    infini::ops::TopkSoftmax::Call(
        handle,
        config,
        planned->x.tensor(planned->x_tensor),
        std::optional<infini::ops::Tensor>{},
        std::optional<infini::ops::Tensor>{},
        planned->norm != 0,
        planned->values.tensor(planned->values_tensor),
        planned->indices.tensor(planned->indices_tensor),
        planned->token_expert_indices.tensor(planned->token_expert_indices_tensor));
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
