#include "infinicore/ops/gelu.hpp"

#ifdef ENABLE_INFINIOPS_API
#include "../infiniops_impl.hpp"

#include "base/gelu_infinilm.h"

#include <string>

namespace infinicore::op::gelu_impl::infiniops {
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
    infini::ops::GeluInfinilm::Call(
        handle,
        config,
        input_meta.tensor(input),
        std::string{"none"},
        output_meta.tensor(output));
}

} // namespace

static bool registered = []() {
    ::infinicore::op::infiniops::registerSupportedDevices(Gelu::dispatcher(), &calculate);
    return true;
}();

} // namespace infinicore::op::gelu_impl::infiniops
#endif
