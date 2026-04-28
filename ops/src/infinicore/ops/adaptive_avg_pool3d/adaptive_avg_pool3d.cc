#include "infinicore/ops/adaptive_avg_pool3d.hpp"
#include <iostream>
#include <stdexcept>
namespace infinicore::op {

common::OpDispatcher<AdaptiveAvgPool3D::schema> &AdaptiveAvgPool3D::dispatcher() {
    static common::OpDispatcher<AdaptiveAvgPool3D::schema> dispatcher_;
    return dispatcher_;
};

void AdaptiveAvgPool3D::execute(Tensor y, Tensor x) {
    auto device_type = context::getDevice().getType();
    auto func = dispatcher().lookup(device_type);
    if (func == nullptr) {
        throw std::runtime_error("No AdaptiveAvgPool3D implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }
    func(y, x);
}

Tensor adaptive_avg_pool3d(Tensor x, std::vector<size_t> output_size) {

    // Create output tensor shap
    Shape y_shape = x->shape();
    y_shape[2] = output_size[0]; // D dimension
    y_shape[3] = output_size[1]; // H dimension
    y_shape[4] = output_size[2]; // W dimension

    auto y = Tensor::empty(y_shape, x->dtype(), x->device());
    adaptive_avg_pool3d_(y, x);
    return y;
}

void adaptive_avg_pool3d_(Tensor y, Tensor x) {
    AdaptiveAvgPool3D::execute(y, x);
}
} // namespace infinicore::op
