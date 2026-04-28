#include "infinicore/ops/cross_entropy.hpp"

#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<CrossEntropy::schema> &CrossEntropy::dispatcher() {
    static common::OpDispatcher<CrossEntropy::schema> dispatcher_;
    return dispatcher_;
};

void CrossEntropy::execute(Tensor output, Tensor input, Tensor target) {

    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(input, target);

    infinicore::context::setDevice(output->device());
    auto device_type = output->device().getType();

    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error("No CrossEntropy implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    func(output, input, target);
}

Tensor cross_entropy(Tensor input, Tensor target) {

    Shape shape = target->shape();

    auto output = Tensor::empty(shape, input->dtype(), input->device());

    cross_entropy_(output, input, target);
    return output;
}

void cross_entropy_(Tensor output, Tensor input, Tensor target) {
    CrossEntropy::execute(output, input, target);
}

} // namespace infinicore::op
