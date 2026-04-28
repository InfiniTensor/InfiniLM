#include "infinicore/ops/gelu.hpp"

#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<Gelu::schema> &Gelu::dispatcher() {
    static common::OpDispatcher<Gelu::schema> dispatcher_;
    return dispatcher_;
};

void Gelu::execute(Tensor output, Tensor input) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);
    infinicore::context::setDevice(output->device());
    auto device_type = output->device().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error("No Gelu implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    func(output, input);
}

Tensor gelu(Tensor input) {
    Shape shape = input->shape();
    auto output = Tensor::empty(shape, input->dtype(), input->device());
    gelu_(output, input);
    return output;
}

void gelu_(Tensor output, Tensor input) {
    Gelu::execute(output, input);
}
} // namespace infinicore::op
