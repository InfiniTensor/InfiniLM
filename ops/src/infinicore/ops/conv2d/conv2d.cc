#include "infinicore/ops/conv2d.hpp"

#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<Conv2d::schema> &Conv2d::dispatcher() {
    static common::OpDispatcher<Conv2d::schema> dispatcher_;
    return dispatcher_;
};

void Conv2d::execute(Tensor output,
                     Tensor input,
                     Tensor weight,
                     Tensor bias,
                     const size_t *pads,
                     const size_t *strides,
                     const size_t *dilations,
                     size_t n) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input, weight, bias);
    infinicore::context::setDevice(output->device());
    auto device_type = output->device().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error("No Conv2d implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    func(output, input, weight, bias, pads, strides, dilations, n);
}

Tensor conv2d(Tensor input,
              Tensor weight,
              Tensor bias,
              const std::vector<size_t> &pads,
              const std::vector<size_t> &strides,
              const std::vector<size_t> &dilations) {
    const auto &in_shape = input->shape(); // [N, C_in, H_in, W_in]
    const auto &w_shape = weight->shape(); // [C_out, C_in, kH, kW]

    // -------------------------------
    // Extract dimensions
    // -------------------------------
    size_t N = in_shape[0];
    size_t C_in = in_shape[1];
    size_t H_in = in_shape[2];
    size_t W_in = in_shape[3];

    size_t C_out = w_shape[0];
    size_t kH = w_shape[2];
    size_t kW = w_shape[3];

    size_t pad_h = pads[0];
    size_t pad_w = pads[1];

    size_t stride_h = strides[0];
    size_t stride_w = strides[1];

    size_t dil_h = dilations[0];
    size_t dil_w = dilations[1];

    auto calc_out = [](size_t in, size_t pad, size_t dilation, size_t kernel, size_t stride) {
        return (in + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1;
    };
    size_t H_out = calc_out(H_in, pad_h, dil_h, kH, stride_h);
    size_t W_out = calc_out(W_in, pad_w, dil_w, kW, stride_w);
    if ((int64_t)H_out <= 0 || (int64_t)W_out <= 0) {
        throw std::runtime_error("Invalid conv2d output shape (negative or zero)");
    }
    Shape out_shape = {N, C_out, H_out, W_out};

    auto output = Tensor::empty(out_shape, input->dtype(), input->device());
    conv2d_(output, input, weight, bias, pads, strides, dilations);
    return output;
}

void conv2d_(Tensor output,
             Tensor input,
             Tensor weight,
             Tensor bias,
             const std::vector<size_t> &pads,
             const std::vector<size_t> &strides,
             const std::vector<size_t> &dilations) {
    if (pads.size() != strides.size() || pads.size() != dilations.size()) {
        throw std::runtime_error("conv2d_: pads/strides/dilations must have the same size");
    }
    Conv2d::execute(output,
                    input,
                    weight,
                    bias,
                    pads.data(),
                    strides.data(),
                    dilations.data(),
                    pads.size());
}
} // namespace infinicore::op
