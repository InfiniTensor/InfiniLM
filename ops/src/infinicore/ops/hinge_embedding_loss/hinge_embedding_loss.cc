#include "infinicore/ops/hinge_embedding_loss.hpp"

#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<HingeEmbeddingLoss::schema> &HingeEmbeddingLoss::dispatcher() {
    static common::OpDispatcher<schema> dispatcher_;
    return dispatcher_;
}

void HingeEmbeddingLoss::execute(
    Tensor output,
    Tensor input,
    Tensor target,
    double margin,
    int reduction) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input, target);
    infinicore::context::setDevice(output->device());
    auto device_type = output->device().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error(
            "No HingeEmbeddingLoss implementation found for device type: "
            + std::to_string(static_cast<int>(device_type)));
    }

    func(output, input, target, margin, reduction);
}

Tensor hinge_embedding_loss(Tensor input, Tensor target, double margin, int reduction) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(input, target);
    INFINICORE_ASSERT(input->dtype() == target->dtype());

    Shape output_shape;
    if (reduction == 0) {
        output_shape = input->shape();
    }

    auto output = Tensor::empty(output_shape, input->dtype(), input->device());
    hinge_embedding_loss_(output, input, target, margin, reduction);
    return output;
}

void hinge_embedding_loss_(
    Tensor output,
    Tensor input,
    Tensor target,
    double margin,
    int reduction) {
    HingeEmbeddingLoss::execute(output, input, target, margin, reduction);
}

} // namespace infinicore::op
