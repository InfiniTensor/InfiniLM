#include "infinicore/ops/tanhshrink.hpp"
#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<Tanhshrink::schema> &Tanhshrink::dispatcher() {
    static common::OpDispatcher<Tanhshrink::schema> dispatcher_;
    return dispatcher_;
};

void Tanhshrink::execute(Tensor output, Tensor input) {
    auto device_type = context::getDevice().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error("No Tanhshrink implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    func(output, input);
}

Tensor tanhshrink(Tensor input) {
    Shape shape = input->shape();
    auto output = Tensor::empty(shape, input->dtype(), input->device());
    tanhshrink_(output, input);
    return output;
}

void tanhshrink_(Tensor output, Tensor input) {
    Tanhshrink::execute(output, input);
}
} // namespace infinicore::op
