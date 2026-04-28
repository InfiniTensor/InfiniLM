#include "infinicore/ops/tan.hpp"
#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<Tan::schema> &Tan::dispatcher() {
    static common::OpDispatcher<Tan::schema> dispatcher_;
    return dispatcher_;
};

void Tan::execute(Tensor output, Tensor input) {
    auto device_type = context::getDevice().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error("No Tan implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    func(output, input);
}

Tensor tan(Tensor input) {
    Shape shape = input->shape();
    auto output = Tensor::empty(shape, input->dtype(), input->device());
    tan_(output, input);
    return output;
}

void tan_(Tensor output, Tensor input) {
    Tan::execute(output, input);
}
} // namespace infinicore::op
