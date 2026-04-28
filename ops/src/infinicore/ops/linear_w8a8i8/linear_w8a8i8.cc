#include "infinicore/ops/linear_w8a8i8.hpp"
#include "infinicore/ops/per_channel_quant_i8.hpp"
#include "infinicore/ops/scaled_mm_i8.hpp"

namespace infinicore::op {

Tensor linear_w8a8i8(Tensor input,
                     Tensor weight_packed,
                     Tensor weight_scale,
                     std::optional<Tensor> bias) {

    // Input is of shape [M, K], Weight_packed is of shape [N, K],stirdes is [N, 1]
    Size ndim = input->ndim();
    Size out_features = weight_packed->shape()[0];

    // Assign memory to out variables
    auto output_shape = input->shape();
    output_shape[ndim - 1] = out_features;
    auto out = Tensor::empty(output_shape, input->dtype(), input->device());

    // Inplace Calculate
    linear_w8a8i8_(out, input, weight_packed, weight_scale, bias);
    return out;
}

void linear_w8a8i8_(Tensor out,
                    Tensor input,
                    Tensor weight_packed,
                    Tensor weight_scale,
                    std::optional<Tensor> bias) {

    auto weight_packed_shape = weight_packed->shape();
    Size out_features = weight_packed_shape[0];
    Size in_features = weight_packed_shape[1];

    Size ndim = input->ndim();
    assert(out->ndim() == ndim);

    Size N = 1;
    auto input_shape = input->shape();
    for (size_t i = 0; i < ndim - 1; ++i) {
        N *= input_shape[i];
    }

    auto input_packed = Tensor::empty(
        {N, input_shape[ndim - 1]},
        DataType::I8,
        input->device());
    auto input_scale = Tensor::empty(
        {N, 1},
        DataType::F32,
        input->device());
    op::per_channel_quant_i8_(input->view({N, in_features}), input_packed, input_scale);
    if (bias.has_value()) {
        bias = std::make_optional(bias.value()->as_strided({N, out_features}, {0, 1}));
    }
    op::scaled_mm_i8_(
        out->view({N, out_features}),
        input_packed,
        input_scale,
        weight_packed->permute({1, 0}),
        weight_scale,
        bias);
}

} // namespace infinicore::op
