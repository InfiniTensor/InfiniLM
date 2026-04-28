#include "infinicore/ops/linear.hpp"
#include "infinicore/ops/gemm.hpp"
#include "infinicore/ops/rearrange.hpp"

#include <algorithm>
#include <cstdlib>

namespace infinicore::op {

Tensor linear(Tensor input,
              Tensor weight,
              std::optional<Tensor> bias) {

    Size ndim = input->ndim();
    Size out_features = weight->shape()[0];

    // Assign memory to out variables
    auto output_shape = input->shape();
    output_shape[ndim - 1] = out_features;
    auto out = Tensor::empty(output_shape, input->dtype(), input->device());

    // Iluvatar workaround: ixblas picks a capture-unfriendly cublas algo when
    // a GEMM's output dim exceeds ~65k (e.g. lm_head with vocab_size ~152k).
    // INFINILM_LINEAR_CHUNK_OUT=N splits any such GEMM along the output dim
    // into chunks of size <= N. Each chunk MUST write to an independent
    // contiguous buffer (cublas selects algo by leading dim, not by shape;
    // writing into a narrow view of the big output keeps ld large and does
    // not bypass the broken algo).
    static const char *chunk_env_ = std::getenv("INFINILM_LINEAR_CHUNK_OUT");
    static const Size chunk_threshold = chunk_env_ ? (Size)std::atoi(chunk_env_) : 0;
    if (chunk_threshold > 0 && out_features > chunk_threshold) {
        for (Size start = 0; start < out_features; start += chunk_threshold) {
            Size len = std::min(chunk_threshold, out_features - start);
            auto weight_chunk = weight->narrow({{0, start, len}});
            std::optional<Tensor> bias_chunk = bias.has_value()
                ? std::make_optional(bias.value()->narrow({{0, start, len}}))
                : std::nullopt;

            auto chunk_shape = input->shape();
            chunk_shape[ndim - 1] = len;
            auto chunk_out = Tensor::empty(chunk_shape, input->dtype(), input->device());

            linear_(chunk_out, input, weight_chunk, bias_chunk);

            auto out_slice = out->narrow({{ndim - 1, start, len}});
            rearrange_(out_slice, chunk_out);
        }
        return out;
    }

    // Inplace Calculate
    linear_(out, input, weight, bias);
    return out;
}

void linear_(Tensor out,
             Tensor input,
             Tensor weight,
             std::optional<Tensor> bias) {

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
    float alpha = 1.0f;
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
