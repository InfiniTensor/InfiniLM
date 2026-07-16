#include "infinicore/ops/linear.hpp"
#include "infinicore/ops/gemm.hpp"
#include "infinicore/ops/rearrange.hpp"

namespace infinicore::op {

Tensor linear(Tensor input,
              Tensor weight,
              std::optional<Tensor> bias,
              float alpha) {

    Size ndim = input->ndim();
    Size out_features = weight->shape()[0];

    // Assign memory to out variables
    auto output_shape = input->shape();
    output_shape[ndim - 1] = out_features;
    auto out = Tensor::empty(output_shape, input->dtype(), input->device());

    // Inplace Calculate
    linear_(out, input, weight, bias, alpha);
    return out;
}

void linear_(Tensor out,
             Tensor input,
             Tensor weight,
             std::optional<Tensor> bias,
             float alpha) {

    auto weight_shape = weight->shape();
    Size out_features = weight_shape[0];
    Size in_features = weight_shape[1];

    Size ndim = input->ndim();
    assert(out->ndim() == ndim);

    // Calculate the number of features
    Size N = 1;
    auto input_shape = input->shape();
    for (size_t i = 0; i < ndim - 1; ++i) {
        N *= input_shape[i];
    }

    // linear transformation
    Tensor out_view = out->view({N, out_features});
    // Add bias
    float beta = 0.0f;
    if (bias.has_value()) {
        rearrange_(out_view,
                   bias.value()->as_strided({N, out_features}, {0, 1}));
        beta = 1.0f;
    }

    gemm_(out_view,
          input->view({N, in_features}),
          weight->permute({1, 0}), alpha, beta);
}

} // namespace infinicore::op
