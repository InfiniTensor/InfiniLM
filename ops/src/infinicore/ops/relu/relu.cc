#include "infinicore/ops/relu.hpp"

#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<Relu::schema> &Relu::dispatcher() {
    static common::OpDispatcher<Relu::schema> dispatcher_;
    return dispatcher_;
};

void Relu::execute(Tensor output, Tensor input) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);
    infinicore::context::setDevice(output->device());
    auto device_type = output->device().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error("No Relu implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    func(output, input);
}

Tensor relu(Tensor input) {
    Shape shape = input->shape();
    auto output = Tensor::empty(shape, input->dtype(), input->device());
    relu_(output, input);
    return output;
}

void relu_(Tensor output, Tensor input) {
    Relu::execute(output, input);
}
} // namespace infinicore::op
