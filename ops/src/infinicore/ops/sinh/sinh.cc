#include "infinicore/ops/sinh.hpp"

#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<Sinh::schema> &Sinh::dispatcher() {
    static common::OpDispatcher<schema> dispatcher_;
    return dispatcher_;
}

void Sinh::execute(Tensor output, Tensor input) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);
    infinicore::context::setDevice(output->device());
    auto device_type = output->device().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error(
            "No Sinh implementation found for device type: "
            + std::to_string(static_cast<int>(device_type)));
    }

    func(output, input);
}

Tensor sinh(Tensor input) {
    Shape shape = input->shape();
    auto output = Tensor::empty(shape, input->dtype(), input->device());
    sinh_(output, input);
    return output;
}

void sinh_(Tensor output, Tensor input) {
    Sinh::execute(output, input);
}

} // namespace infinicore::op
