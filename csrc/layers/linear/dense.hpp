#pragma once

#include "../../execution/submit.hpp"
#include <base/gemm.h>
#include <optional>
#include <stdexcept>

namespace infinilm::layers::linear {

// Dense-layer policy: weight layout, output dtype, and bias composition.
// Ops owns GEMM/Add mathematics, dispatch, precision, and workspace.
inline infinilm::runtime::DataType resolve_output_dtype(
    const infinilm::runtime::Device &device, infinilm::runtime::DataType input_dtype,
    infinilm::runtime::DataType weight_dtype, bool has_bias,
    std::optional<infinilm::runtime::DataType> output_dtype) {
    if (weight_dtype != input_dtype)
        throw std::invalid_argument("linear input and weight dtypes differ");
    const auto resolved = output_dtype.value_or(input_dtype);
    return resolved;
}

inline void validate_projection(
    infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &input,
    const infinilm::runtime::Tensor &weight, const std::optional<infinilm::runtime::Tensor> &bias,
    infinilm::runtime::DataType output_dtype, bool packed) {
    runtime.assertOwnerThread();
    if (!input || !weight || (bias && !*bias))
        throw std::invalid_argument("linear tensors must be nonnull");
    if (input->ndim() == 0 || weight->ndim() != 2)
        throw std::invalid_argument("linear requires non-scalar input and a matrix weight");
    if (input->shape().back() != weight->size(packed ? 0 : 1) ||
        input->device() != runtime.device() || weight->device() != runtime.device())
        throw std::invalid_argument("linear views or runtime do not match");
    (void)resolve_output_dtype(runtime.device(), input->dtype(), weight->dtype(),
                               bias.has_value(), output_dtype);
    if (bias && ((*bias)->ndim() != 1 || (*bias)->size(0) != weight->size(packed ? 1 : 0) ||
                 (*bias)->dtype() != output_dtype || (*bias)->device() != runtime.device()))
        throw std::invalid_argument("linear bias must match output features, dtype and device");
}

// A layer combination, with an optional caller-provided output for router/
// Python inplace consumers. Packed weights retain their existing [IC, OC] view.
inline infinilm::runtime::Tensor dense_projection(
    infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &input,
    const infinilm::runtime::Tensor &weight, const std::optional<infinilm::runtime::Tensor> &bias,
    float alpha = 1.0F, std::optional<infinilm::runtime::DataType> output_dtype = std::nullopt,
    bool packed = false, infinilm::runtime::Tensor output = {}) {
    const auto dtype = output ? output->dtype() :
        output_dtype.value_or(input ? input->dtype() : infinilm::runtime::DataType::kFloat32);
    validate_projection(runtime, input, weight, bias, dtype, packed);
    auto shape = input->shape();
    shape.back() = weight->size(packed ? 1 : 0);
    if (output && (output->shape() != shape || output->device() != runtime.device()))
        throw std::invalid_argument("linear views or runtime do not match");
    if (!output) output = infinilm::runtime::Tensor::empty(runtime, shape, dtype);
    const auto packed_weight = packed ? weight : weight->permute({1, 0});
    infinilm::runtime::Size rows = 1;
    for (size_t dim = 0; dim + 1 < input->ndim(); ++dim) rows *= input->size(dim);
    const auto input_view = input->view({rows, packed_weight->size(0)});
    const auto output_view = output->view({rows, packed_weight->size(1)});
    const auto device = runtime.device().type();
    infinilm::execution::submit<infini::ops::Gemm>(
        runtime, infinilm::execution::RecordingPolicy::DeviceOnly,
        infini::ops::Gemm::DefaultConfig(device), input_view, packed_weight,
        std::optional<infinilm::runtime::Tensor>{}, std::optional<float>{alpha},
        std::optional<float>{0.0F}, std::optional<int>{}, std::optional<int>{}, output_view);
    INFINILM_TRACE_OP(Gemm, output_view);
    if (bias) {
        infinilm::execution::submit<infini::ops::Add>(
            runtime, infini::ops::Add::DefaultConfig(device), output_view, *bias, output_view);
        INFINILM_TRACE_OP(Add, output_view);
    }
    return output;
}

} // namespace infinilm::layers::linear
