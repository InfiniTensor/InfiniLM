#include "infinicore/ops/kron.hpp"

#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<Kron::schema> &Kron::dispatcher() {
    static common::OpDispatcher<schema> dispatcher_;
    return dispatcher_;
}

void Kron::execute(Tensor output, Tensor a, Tensor b) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, a, b);
    infinicore::context::setDevice(output->device());
    auto device_type = output->device().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error(
            "No Kron implementation found for device type: "
            + std::to_string(static_cast<int>(device_type)));
    }

    func(output, a, b);
}

Tensor kron(Tensor a, Tensor b) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(a, b);
    INFINICORE_ASSERT(a->dtype() == b->dtype());
    INFINICORE_ASSERT(a->shape().size() == b->shape().size());

    const auto &a_shape = a->shape();
    const auto &b_shape = b->shape();
    Shape y_shape(a_shape.size());
    for (size_t i = 0; i < a_shape.size(); ++i) {
        y_shape[i] = a_shape[i] * b_shape[i];
    }

    auto output = Tensor::empty(y_shape, a->dtype(), a->device());
    kron_(output, a, b);
    return output;
}

void kron_(Tensor output, Tensor a, Tensor b) {
    Kron::execute(output, a, b);
}

} // namespace infinicore::op
