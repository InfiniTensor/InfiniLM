#include "infinicore/ops/conv2d.hpp"

#ifdef ENABLE_INFINIOPS_API
#include "../infiniops_impl.hpp"

#include "base/conv_infinilm.h"

#include <optional>
#include <vector>

namespace infinicore::op::conv2d_impl::infiniops {
namespace {

using TensorMeta = ::infinicore::op::infiniops::TensorMeta;

std::vector<int64_t> toInt64Vector(const size_t *values, size_t n) {
    std::vector<int64_t> result;
    result.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        result.push_back(static_cast<int64_t>(values[i]));
    }
    return result;
}

void calculate(Tensor output,
               Tensor input,
               Tensor weight,
               Tensor bias,
               const size_t *pads,
               const size_t *strides,
               const size_t *dilations,
               size_t n) {
    INFINICORE_ASSERT(::infinicore::op::infiniops::isSupportedDevice(output->device().type()));
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input, weight);
    if (bias) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, bias);
    }

    infini::ops::Handle handle;
    handle.set_stream(context::getStream());
    infini::ops::Config config;

    TensorMeta output_meta(output);
    TensorMeta input_meta(input);
    TensorMeta weight_meta(weight);
    std::optional<TensorMeta> bias_meta;
    if (bias) {
        bias_meta.emplace(bias);
    }

    infini::ops::ConvInfinilm::Call(
        handle,
        config,
        input_meta.tensor(input),
        weight_meta.tensor(weight),
        bias_meta ? std::optional<infini::ops::Tensor>{bias_meta->tensor(bias)} : std::nullopt,
        toInt64Vector(pads, n),
        toInt64Vector(strides, n),
        toInt64Vector(dilations, n),
        int64_t{1},
        output_meta.tensor(output));
}

} // namespace

static bool registered = []() {
    ::infinicore::op::infiniops::registerSupportedDevices(Conv2d::dispatcher(), &calculate);
    return true;
}();

} // namespace infinicore::op::conv2d_impl::infiniops
#endif
