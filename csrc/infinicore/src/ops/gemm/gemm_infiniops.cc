#include "infinicore/ops/gemm.hpp"

#ifdef ENABLE_INFINIOPS_API
#include "../infiniops_impl.hpp"

#include <optional>

namespace infinicore::op::gemm_impl::infiniops {
namespace {

using TensorMeta = ::infinicore::op::infiniops::TensorMeta;

struct PlannedMeta {
    TensorMeta c, a, b;
    graph::GraphTensor c_tensor, a_tensor, b_tensor;
    float alpha, beta;
};

} // namespace

void *plan(Tensor c, const Tensor &a, const Tensor &b, float alpha, float beta) {
    INFINICORE_ASSERT(::infinicore::op::infiniops::isSupportedDevice(c->device().type()));
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(c, a, b);

    return new PlannedMeta{
        TensorMeta(c),
        TensorMeta(a),
        TensorMeta(b),
        graph::GraphTensor(c),
        graph::GraphTensor(a),
        graph::GraphTensor(b),
        alpha,
        beta};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    infini::ops::Handle handle;
    handle.set_stream(context::getStream());
    infini::ops::Config config;

    infini::ops::Gemm::Call(
        handle,
        config,
        planned->a.tensor(planned->a_tensor),
        planned->b.tensor(planned->b_tensor),
        std::optional<float>{planned->alpha},
        std::optional<float>{planned->beta},
        std::optional<int>{},
        std::optional<int>{},
        planned->c.tensor(planned->c_tensor));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    ::infinicore::op::infiniops::registerSupportedDevices(Gemm::plan_dispatcher(), &plan);
    ::infinicore::op::infiniops::registerSupportedDevices(Gemm::run_dispatcher(), &run);
    ::infinicore::op::infiniops::registerSupportedDevices(Gemm::cleanup_dispatcher(), &cleanup);
    return true;
}();

} // namespace infinicore::op::gemm_impl::infiniops
#endif
