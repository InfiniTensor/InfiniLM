#include "infinicore/ops/swiglu.hpp"

#ifdef ENABLE_INFINIOPS_API
#include "../infiniops_impl.hpp"

#include "base/copy.h"
#include "base/silu_and_mul.h"

namespace infinicore::op::swiglu_impl::infiniops {
namespace {

using TensorMeta = ::infinicore::op::infiniops::TensorMeta;

struct PlannedMeta {
    TensorMeta c, a, b, packed, gate, up;
    graph::GraphTensor c_tensor, a_tensor, b_tensor, packed_tensor, gate_tensor, up_tensor;
    Tensor packed_owner;
};

} // namespace

void *plan(Tensor c, const Tensor &a, const Tensor &b) {
    INFINICORE_ASSERT(::infinicore::op::infiniops::isSupportedDevice(c->device().type()));
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(c, a, b);
    INFINICORE_ASSERT(c->shape() == a->shape());
    INFINICORE_ASSERT(a->shape() == b->shape());
    INFINICORE_ASSERT(c->dtype() == a->dtype());
    INFINICORE_ASSERT(a->dtype() == b->dtype());
    INFINICORE_ASSERT(!a->shape().empty());

    auto packed_shape = a->shape();
    packed_shape.back() *= 2;
    auto packed = Tensor::empty(packed_shape, a->dtype(), a->device());
    auto hidden_size = a->size(a->ndim() - 1);
    auto gate = packed->narrow({{packed->ndim() - 1, 0, hidden_size}});
    auto up = packed->narrow({{packed->ndim() - 1, hidden_size, hidden_size}});

    return new PlannedMeta{
        TensorMeta(c),
        TensorMeta(a),
        TensorMeta(b),
        TensorMeta(packed),
        TensorMeta(gate),
        TensorMeta(up),
        graph::GraphTensor(c),
        graph::GraphTensor(a),
        graph::GraphTensor(b),
        graph::GraphTensor(packed),
        graph::GraphTensor(gate),
        graph::GraphTensor(up),
        packed};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    infini::ops::Handle handle;
    handle.set_stream(context::getStream());
    infini::ops::Config config;

    infini::ops::Copy::Call(
        handle,
        config,
        planned->b.tensor(planned->b_tensor),
        false,
        planned->gate.tensor(planned->gate_tensor));
    infini::ops::Copy::Call(
        handle,
        config,
        planned->a.tensor(planned->a_tensor),
        false,
        planned->up.tensor(planned->up_tensor));
    infini::ops::SiluAndMul::Call(
        handle,
        config,
        planned->packed.tensor(planned->packed_tensor),
        planned->c.tensor(planned->c_tensor));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    ::infinicore::op::infiniops::registerSupportedDevices(SwiGLU::plan_dispatcher(), &plan);
    ::infinicore::op::infiniops::registerSupportedDevices(SwiGLU::run_dispatcher(), &run);
    ::infinicore::op::infiniops::registerSupportedDevices(SwiGLU::cleanup_dispatcher(), &cleanup);
    return true;
}();

} // namespace infinicore::op::swiglu_impl::infiniops
#endif
