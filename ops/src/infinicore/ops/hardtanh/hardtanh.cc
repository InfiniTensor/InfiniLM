#include "infinicore/ops/hardtanh.hpp"

#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<HardTanh::schema> &HardTanh::dispatcher() {
    static common::OpDispatcher<HardTanh::schema> dispatcher_;
    return dispatcher_;
}

void HardTanh::execute(Tensor output, Tensor input, float min_val, float max_val) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);
    infinicore::context::setDevice(output->device());

    auto device_type = output->device().getType();
    auto func = dispatcher().lookup(device_type);
    if (func == nullptr) {
        throw std::runtime_error(
            "No HardTanh implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    func(output, input, min_val, max_val);
}

Tensor hardtanh(Tensor input, float min_val, float max_val) {
    auto output = Tensor::empty(input->shape(), input->dtype(), input->device());
    hardtanh_(output, input, min_val, max_val);
    return output;
}

void hardtanh_(Tensor output, Tensor input, float min_val, float max_val) {
    HardTanh::execute(output, input, min_val, max_val);
}

} // namespace infinicore::op
