#include "infinicore/ops/silu.hpp"

#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<Silu::schema> &Silu::dispatcher() {
    static common::OpDispatcher<Silu::schema> dispatcher_;
    return dispatcher_;
};

void Silu::execute(Tensor output, Tensor input) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);
    infinicore::context::setDevice(output->device());
    auto device_type = output->device().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error("No Silu implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    func(output, input);
}

Tensor silu(Tensor input) {
    Shape shape = input->shape();
    auto output = Tensor::empty(shape, input->dtype(), input->device());
    silu_(output, input);
    return output;
}

void silu_(Tensor output, Tensor input) {
    Silu::execute(output, input);
}
} // namespace infinicore::op
