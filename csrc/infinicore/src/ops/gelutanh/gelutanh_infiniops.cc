#include "infinicore/ops/gelutanh.hpp"

#ifdef ENABLE_INFINIOPS_API
#include "../infiniops_impl.hpp"

#include "base/gelutanh_infinilm.h"

namespace infinicore::op::gelutanh_impl::infiniops {
namespace {

using TensorMeta = ::infinicore::op::infiniops::TensorMeta;

void calculate(Tensor output, Tensor input) {
    INFINICORE_ASSERT(::infinicore::op::infiniops::isSupportedDevice(output->device().type()));
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);

    infini::ops::Handle handle;
    handle.set_stream(context::getStream());
    infini::ops::Config config;

    TensorMeta output_meta(output);
    TensorMeta input_meta(input);
    infini::ops::GelutanhInfinilm::Call(
        handle,
        config,
        input_meta.tensor(input),
        output_meta.tensor(output));
}

} // namespace

static bool registered = []() {
    ::infinicore::op::infiniops::registerSupportedDevices(GeluTanh::dispatcher(), &calculate);
    return true;
}();

} // namespace infinicore::op::gelutanh_impl::infiniops
#endif
