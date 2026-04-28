#include "infinicore/ops/gelutanh.hpp"

#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<GeluTanh::schema> &GeluTanh::dispatcher() {
    static common::OpDispatcher<GeluTanh::schema> dispatcher_;
    return dispatcher_;
};

void GeluTanh::execute(Tensor output, Tensor input) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);
    infinicore::context::setDevice(output->device());
    auto device_type = output->device().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error("No GeluTanh implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    func(output, input);
}

Tensor gelu_tanh(Tensor input) {
    Shape shape = input->shape();
    auto output = Tensor::empty(shape, input->dtype(), input->device());
    gelu_tanh_(output, input);
    return output;
}

void gelu_tanh_(Tensor output, Tensor input) {
    GeluTanh::execute(output, input);
}
} // namespace infinicore::op
