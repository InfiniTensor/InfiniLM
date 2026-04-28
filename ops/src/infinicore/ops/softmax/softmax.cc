#include "infinicore/ops/softmax.hpp"

#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<Softmax::schema> &Softmax::dispatcher() {
    static common::OpDispatcher<Softmax::schema> dispatcher_;
    return dispatcher_;
};

void Softmax::execute(Tensor output, Tensor input, int axis) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);
    infinicore::context::setDevice(output->device());
    auto device_type = output->device().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error("No Softmax implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    func(output, input, axis);
}

Tensor softmax(Tensor input, int axis) {
    Shape shape = input->shape();
    auto output = Tensor::empty(shape, input->dtype(), input->device());
    softmax_(output, input, axis);
    return output;
}

void softmax_(Tensor output, Tensor input, int axis) {
    Softmax::execute(output, input, axis);
}
} // namespace infinicore::op
