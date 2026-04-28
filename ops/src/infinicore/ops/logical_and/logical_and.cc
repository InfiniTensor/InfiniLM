#include "infinicore/ops/logical_and.hpp"
#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<LogicalAnd::schema> &LogicalAnd::dispatcher() {
    static common::OpDispatcher<LogicalAnd::schema> dispatcher_;
    return dispatcher_;
};

void LogicalAnd::execute(Tensor output, Tensor input1, Tensor input2) {
    // --- 修正点：去掉第二个参数 true ---
    infinicore::context::setDevice(input1->device());

    auto device_type = context::getDevice().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error("No LogicalAnd implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    func(output, input1, input2);
}

Tensor logical_and(Tensor input1, Tensor input2) {
    Shape shape = input1->shape();
    auto output = Tensor::empty(shape, input1->dtype(), input1->device());

    logical_and_(output, input1, input2);
    return output;
}

void logical_and_(Tensor output, Tensor input1, Tensor input2) {
    LogicalAnd::execute(output, input1, input2);
}

} // namespace infinicore::op
