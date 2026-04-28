#include "infinicore/ops/adaptive_avg_pool1d.hpp"
#include <stdexcept>
#include <vector>

namespace infinicore::op {

common::OpDispatcher<AdaptiveAvgPool1d::schema> &AdaptiveAvgPool1d::dispatcher() {
    static common::OpDispatcher<AdaptiveAvgPool1d::schema> dispatcher_;
    return dispatcher_;
};

void AdaptiveAvgPool1d::execute(Tensor output, Tensor input) {
    auto device_type = context::getDevice().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error("No AdaptiveAvgPool1d implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    func(output, input);
}

Tensor adaptive_avg_pool1d(Tensor input, int64_t output_size) {
    size_t ndim = input->ndim();
    if (ndim != 2 && ndim != 3) {
        throw std::runtime_error("AdaptiveAvgPool1d: Input tensor must be 2D or 3D.");
    }

    if (output_size <= 0) {
        throw std::runtime_error("AdaptiveAvgPool1d: output_size must be positive.");
    }

    auto out_shape = input->shape();
    out_shape[ndim - 1] = output_size;

    auto output = Tensor::empty(out_shape, input->dtype(), input->device());

    AdaptiveAvgPool1d::execute(output, input);

    return output;
}

} // namespace infinicore::op
