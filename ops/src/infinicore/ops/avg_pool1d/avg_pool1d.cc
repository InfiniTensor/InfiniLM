#include "infinicore/ops/avg_pool1d.hpp"

#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<AvgPool1d::schema> &AvgPool1d::dispatcher() {
    static common::OpDispatcher<AvgPool1d::schema> dispatcher_;
    return dispatcher_;
}

void AvgPool1d::execute(
    Tensor output,
    Tensor input,
    size_t kernel_size,
    size_t stride,
    size_t padding) {

    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);
    if (stride == 0) {
        stride = kernel_size;
    }

    infinicore::context::setDevice(output->device());
    auto device_type = output->device().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error(
            "No AvgPool1d implementation for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    func(output, input, kernel_size, stride, padding);
}

Tensor avg_pool1d(Tensor input, size_t kernel_size, size_t stride, size_t padding) {
    if (stride == 0) {
        stride = kernel_size;
    }

    const auto &shape = input->shape();
    if (shape.size() != 3) {
        throw std::runtime_error("AvgPool1d expects tensors with shape [N, C, L]");
    }

    const size_t n = shape[0];
    const size_t c = shape[1];
    const size_t l_in = shape[2];

    if (l_in + 2 * padding < kernel_size) {
        throw std::runtime_error("AvgPool1d kernel_size is larger than padded length");
    }

    const size_t out_width = (l_in + 2 * padding - kernel_size) / stride + 1;

    Shape out_shape = {n, c, out_width};
    auto output = Tensor::empty(out_shape, input->dtype(), input->device());
    avg_pool1d_(output, input, kernel_size, stride, padding);
    return output;
}

void avg_pool1d_(Tensor output, Tensor input, size_t kernel_size, size_t stride, size_t padding) {
    AvgPool1d::execute(output, input, kernel_size, stride, padding);
}

} // namespace infinicore::op
