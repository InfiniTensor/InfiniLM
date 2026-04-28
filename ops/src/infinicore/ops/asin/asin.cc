#include "infinicore/ops/asin.hpp"

#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<Asin::schema> &Asin::dispatcher() {
    static common::OpDispatcher<Asin::schema> dispatcher_;
    return dispatcher_;
};
void Asin::execute(Tensor output, Tensor input) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);
    infinicore::context::setDevice(output->device());
    auto device_type = output->device().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error("No Asin implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    func(output, input);
}

Tensor asin(Tensor input) {
    Shape shape = input->shape();
    auto output = Tensor::empty(shape, input->dtype(), input->device());
    asin_(output, input);
    return output;
}

void asin_(Tensor output, Tensor input) {
    Asin::execute(output, input);
}
} // namespace infinicore::op
