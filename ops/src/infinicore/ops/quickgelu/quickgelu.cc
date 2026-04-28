#include "infinicore/ops/quickgelu.hpp"

#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<QuickGelu::schema> &QuickGelu::dispatcher() {
    static common::OpDispatcher<QuickGelu::schema> dispatcher_;
    return dispatcher_;
};

void QuickGelu::execute(Tensor output, Tensor input) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);
    infinicore::context::setDevice(output->device());
    auto device_type = output->device().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error("No QuickGelu implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    func(output, input);
}

Tensor quick_gelu(Tensor input) {
    Shape shape = input->shape();
    auto output = Tensor::empty(shape, input->dtype(), input->device());
    quick_gelu_(output, input);
    return output;
}

void quick_gelu_(Tensor output, Tensor input) {
    QuickGelu::execute(output, input);
}
} // namespace infinicore::op
