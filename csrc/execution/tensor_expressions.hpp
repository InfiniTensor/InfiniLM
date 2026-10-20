#pragma once

#include "tensor_checks.hpp"

namespace infinilm::execution {

// Inferred shape is used by the existing variable-length VideoNSA joins.
inline infinilm::runtime::Tensor cat_alloc(infinilm::runtime::Runtime &runtime,
                                           std::vector<infinilm::runtime::Tensor> inputs,
                                           int dim) {
    const auto layout = tensor_checks::cat::analyze(runtime, inputs, dim);
    auto shape = (*layout.reference)->shape();
    shape[layout.axis] = layout.length;
    auto output = infinilm::runtime::Tensor::empty(runtime, shape, (*layout.reference)->dtype());
    tensor_checks::cat::validate(runtime, output, inputs, layout);
    const auto axis = layout.axis;
    inputs.erase(std::remove_if(inputs.begin(), inputs.end(), tensor_checks::cat::empty_vector),
                 inputs.end());
    if (output->numel()) {
        submit<infini::ops::Cat>(runtime, infini::ops::Cat::DefaultConfig(runtime.device().type()),
                                 inputs, static_cast<int64_t>(axis), output);
    }
    return output;
}

// Only the existing fixed Q/K pairs pass a model-selected shape and positive axis.
// Their allocation precedes validation, as it did at each model call site.
inline infinilm::runtime::Tensor cat_alloc(infinilm::runtime::Runtime &runtime,
                                           const std::vector<infinilm::runtime::Tensor> &inputs,
                                           size_t axis,
                                           const infinilm::runtime::Shape &output_shape) {
    auto output = infinilm::runtime::Tensor::empty(runtime, output_shape, inputs[0]->dtype());
    tensor_checks::cat::validate(runtime, output, inputs, {nullptr, axis, 0});
    if (output->numel()) {
        submit<infini::ops::Cat>(runtime, infini::ops::Cat::DefaultConfig(runtime.device().type()),
                                 inputs, static_cast<int64_t>(axis), output);
    }
    return output;
}

inline infinilm::runtime::Tensor pad_constant_alloc(infinilm::runtime::Runtime &runtime,
                                                    const infinilm::runtime::Tensor &input,
                                                    const std::vector<int> &padding,
                                                    double value) {
    const auto layout = tensor_checks::pad::analyze(input, padding);
    auto output = infinilm::runtime::Tensor::empty(runtime, layout.output_shape, input->dtype());
    if (output->numel()) {
        submit<infini::ops::Pad>(
            runtime, infini::ops::Pad::DefaultConfig(runtime.device().type()),
            input, std::vector<int64_t>(padding.begin(), padding.end()),
            std::string{"constant"}, std::optional<double>{value}, output);
    }
    return output;
}

inline infinilm::runtime::Tensor broadcast_materialized(infinilm::runtime::Runtime &runtime,
                                                        const infinilm::runtime::Tensor &input,
                                                        const std::vector<int64_t> &target_shape) {
    const auto shape = tensor_checks::broadcast_to::checked_shape(target_shape, input->element_size());
    tensor_checks::broadcast_to::validate_shape(input->shape(), shape);
    auto output = infinilm::runtime::Tensor::empty(runtime, shape, input->dtype());
    if (output->numel()) {
        submit<infini::ops::Copy>(runtime, infini::ops::Copy::DefaultConfig(runtime.device().type()),
                                  input, false, output);
    }
    return output;
}

} // namespace infinilm::execution
