#include "infinicore/ops/selu.hpp"

#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<Selu::schema> &Selu::dispatcher() {
    static common::OpDispatcher<schema> dispatcher_;
    return dispatcher_;
}

void Selu::execute(Tensor output, Tensor input) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);
    infinicore::context::setDevice(output->device());
    auto device_type = output->device().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error(
            "No Selu implementation found for device type: "
            + std::to_string(static_cast<int>(device_type)));
    }

    func(output, input);
}

Tensor selu(Tensor input) {
    Shape shape = input->shape();
    auto output = Tensor::empty(shape, input->dtype(), input->device());
    selu_(output, input);
    return output;
}

void selu_(Tensor output, Tensor input) {
    Selu::execute(output, input);
}

} // namespace infinicore::op
