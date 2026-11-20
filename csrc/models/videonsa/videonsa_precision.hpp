#pragma once
#include "../../layers/linear/linear.hpp"
#include "../../layers/linear/dense.hpp"
#include "../../execution/submit.hpp"
namespace infinilm::models::videonsa {
// VideoNSA checkpoint projections historically add bias before BF16 rounding.
// Keep this policy local; other models retain the shared dense-layer policy.
inline runtime::Tensor precise_biased_linear(runtime::Runtime &runtime,
                                             const nn::Linear &layer,
                                             runtime::Tensor &input) {
    if (runtime.device().type() != runtime::Device::Type::kHygon
        || input->dtype() != runtime::DataType::kBFloat16
        || layer.output_dtype() != runtime::DataType::kBFloat16 || !layer.has_bias()
        || !dynamic_cast<quantization::NoneQuantization *>(layer.get_quantization().get())) {
        return layer.forward(runtime, input);
    }
    auto weight = layer.weight();
    auto bias = layer.bias();
    layers::linear::validate_projection(runtime, input, weight, bias, input->dtype(), false);
    auto gemm_input = input;
    auto gemm_weight = weight;
    // The BW BF16->FP32 BLAS path for this nonaligned vision-MLP shape
    // produced different results for identical inputs. FP32 operands select
    // the repeatable native path, without changing the shared GEMM policy.
    if (input->numel() / input->shape().back() == 256
        && weight->shape() == runtime::Shape{3420, 1280}) {
        gemm_input = runtime::Tensor::empty(runtime, input->shape(), runtime::DataType::kFloat32);
        gemm_weight = runtime::Tensor::empty(runtime, weight->shape(), runtime::DataType::kFloat32);
        auto config = infini::ops::Cast::DefaultConfig(runtime.device().type());
        execution::submit<infini::ops::Cast>(runtime, execution::RecordingPolicy::DeviceOnly,
            config, input, gemm_input);
        execution::submit<infini::ops::Cast>(runtime, execution::RecordingPolicy::DeviceOnly,
            config, weight, gemm_weight);
    }
    auto sum = layers::linear::dense_projection(runtime, gemm_input, gemm_weight, std::nullopt,
        layer.alpha(), runtime::DataType::kFloat32);
    auto bias32 = runtime::Tensor::empty(runtime, bias->shape(), runtime::DataType::kFloat32);
    execution::submit<infini::ops::Cast>(runtime, execution::RecordingPolicy::DeviceOnly,
        infini::ops::Cast::DefaultConfig(runtime.device().type()), bias, bias32);
    execution::submit<infini::ops::Add>(runtime, execution::RecordingPolicy::DeviceOnly,
        infini::ops::Add::DefaultConfig(runtime.device().type()), sum, bias32, sum);
    auto out = runtime::Tensor::empty(runtime, sum->shape(), input->dtype());
    execution::submit<infini::ops::Cast>(runtime, execution::RecordingPolicy::DeviceOnly,
        infini::ops::Cast::DefaultConfig(runtime.device().type()), sum, out);
    return out;
}
} // namespace infinilm::models::videonsa
