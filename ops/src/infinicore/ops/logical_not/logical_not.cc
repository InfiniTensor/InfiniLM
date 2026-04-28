#include "infinicore/ops/logical_not.hpp"
#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<LogicalNot::schema> &
LogicalNot::dispatcher() {
    static common::OpDispatcher<LogicalNot::schema> dispatcher_;
    return dispatcher_;
}

void LogicalNot::execute(Tensor output, Tensor input) {
    infinicore::context::setDevice(input->device());

    auto device_type = context::getDevice().getType();
    auto func = dispatcher().lookup(device_type);
    if (func == nullptr) {
        throw std::runtime_error(
            "No LogicalNot implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    func(output, input);
}

Tensor logical_not(Tensor input) {
    auto output = Tensor::empty(input->shape(), input->dtype(), input->device());
    logical_not_(output, input);
    return output;
}

void logical_not_(Tensor output, Tensor input) {
    LogicalNot::execute(output, input);
}

} // namespace infinicore::op
