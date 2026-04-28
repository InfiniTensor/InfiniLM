#include "infinicore/ops/hardswish.hpp"

#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<Hardswish::schema> &Hardswish::dispatcher() {
    static common::OpDispatcher<Hardswish::schema> dispatcher_;
    return dispatcher_;
}

void Hardswish::execute(Tensor output, Tensor input) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);
    infinicore::context::setDevice(output->device());
    auto device_type = output->device().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error(
            "No Hardswish implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    func(output, input);
}

Tensor hardswish(Tensor input) {
    auto output = Tensor::empty(input->shape(), input->dtype(), input->device());
    hardswish_(output, input);
    return output;
}

void hardswish_(Tensor output, Tensor input) {
    Hardswish::execute(output, input);
}

} // namespace infinicore::op
