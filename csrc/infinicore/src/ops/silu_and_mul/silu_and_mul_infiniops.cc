#include "infinicore/ops/silu_and_mul.hpp"

#ifdef ENABLE_INFINIOPS_API
#include "../infiniops_impl.hpp"

#include "base/silu_and_mul_infinilm.h"

namespace infinicore::op::silu_and_mul_impl::infiniops {
namespace {
using TensorMeta = ::infinicore::op::infiniops::TensorMeta;
struct PlannedMeta {
    TensorMeta output, input;
    graph::GraphTensor output_tensor, input_tensor;
};
} // namespace

void *plan(Tensor output, const Tensor &input) {
    INFINICORE_ASSERT(::infinicore::op::infiniops::isSupportedDevice(output->device().type()));
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);
    return new PlannedMeta{TensorMeta(output), TensorMeta(input), graph::GraphTensor(output), graph::GraphTensor(input)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    infini::ops::Handle handle;
    handle.set_stream(context::getStream());
    infini::ops::Config config;
    infini::ops::SiluAndMulInfinilm::Call(
        handle, config, planned->input.tensor(planned->input_tensor), planned->output.tensor(planned->output_tensor));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    ::infinicore::op::infiniops::registerSupportedDevices(SiluAndMul::plan_dispatcher(), &plan);
    ::infinicore::op::infiniops::registerSupportedDevices(SiluAndMul::run_dispatcher(), &run);
    ::infinicore::op::infiniops::registerSupportedDevices(SiluAndMul::cleanup_dispatcher(), &cleanup);
    return true;
}();
} // namespace infinicore::op::silu_and_mul_impl::infiniops
#endif
